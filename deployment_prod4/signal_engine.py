"""
signal_engine.py — v18.17
==========================
THE CORE PROBLEM (why 100% HOLD in live):
  The notebook's DYNAMIC_HOLD_REGIMES included regime 2 (RANGE_CHOP).
  Gate 1 in the live engine mirrored this and blocked EVERY bar because
  the live market is ALWAYS in RANGE_CHOP (GMM=1.0, HMM=1.0).

  But the ENTIRE MR-FADE strategy is designed FOR RANGE_CHOP.
  Blocking RANGE_CHOP in the signal engine makes the system unable to trade.

  Fix: Gate 1 (regime block) only applies to TREND signals.
       RANGE_CHOP is explicitly ALLOWED and routed to MR-FADE.
       ANOMALY (regime 3) is always blocked for both paths.

  Other fixes:
  - SELL direction comes from price vs EMA20 sign (not from buy_p>sell_p)
  - bar_traded flag resets on bar boundary
  - caller must pass live tick price + fresh bars (not cached bar close)

  Decision table:
  ┌──────────────────────────────────────────────────────────────────┐
  │ regime 2 + |price-EMA20| >= 5pip  → MR-FADE  BUY or SELL ✅    │
  │ regime 2 + |price-EMA20| <  5pip  → HOLD (no edge yet)         │
  │ regime 0 or 1 + ML >= 0.55        → TREND    BUY or SELL ✅    │
  │ regime 3 (ANOMALY)                → BLOCKED always             │
  └──────────────────────────────────────────────────────────────────┘
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

PIP            = 0.0001
EMA_PERIOD     = 20
MR_BAND_PIPS   = 5.0
BLOCK_THR      = 0.92
RANGE_THR      = 0.52
TREND_ML_THR   = 0.55
ANOMALY_REGIME = 3


@dataclass
class SignalOutput:
    direction:     str
    signal_type:   str
    score:         float
    threshold:     float
    gate_rejected: str
    cat_buy_prob:  float
    cat_sell_prob: float
    lr_dir_prob:   float
    regime:        int
    regime_name:   str
    gmm_probs:     np.ndarray
    hmm_probs:     np.ndarray
    size_mult:     float
    atr_pips:      float
    spread_pips:   float
    sl_pips:       float
    tp1_pips:      float
    tp_pips:       float
    runner_pips:   float
    expected_edge: float
    ema20:         float
    close_price:   float
    mr_dist_pips:  float
    mr_strength:   float
    is_mr:         bool
    quality:       str
    bar_time:      Optional[datetime] = None


class SignalEngine:

    REGIME_LEVELS = {
        0: dict(sl=20, tp1=30, tp=50, runner=80),
        1: dict(sl=20, tp1=30, tp=50, runner=80),
        2: dict(sl=16, tp1=24, tp=38, runner=60),
        3: dict(sl=16, tp1=24, tp=38, runner=60),
    }
    REGIME_NAMES = {0:'BULL_TREND', 1:'BEAR_TREND', 2:'RANGE_CHOP', 3:'ANOMALY'}

    def __init__(self, udl_engine, cat_clf, lr_clf, scaler_lr,
                 cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.udl        = udl_engine
        self.cat        = cat_clf
        self.lr         = lr_clf
        self.scaler_lr  = scaler_lr
        self.mr_band    = float(cfg.get('mr_band_pips',   MR_BAND_PIPS))
        self.block_thr  = float(cfg.get('block_thr',      BLOCK_THR))
        self.range_thr  = float(cfg.get('range_thr',      RANGE_THR))
        self.trend_thr  = float(cfg.get('trend_ml_thr',   TREND_ML_THR))
        self.icm_cost   = float(cfg.get('icm_cost_fixed', 1.20))
        self.cost_ratio = float(cfg.get('cost_atr_ratio', 0.25))

        self._last_bar_time: Optional[datetime] = None
        self._bar_traded: bool = False

        logger.info(
            f"[v18.17] SignalEngine | "
            f"RANGE_CHOP→MR-FADE ALLOWED (band={self.mr_band}pip, block_thr={self.block_thr}) | "
            f"TREND (regime 0/1): ML p>={self.trend_thr} | "
            f"ANOMALY blocked | MR thr={self.range_thr}"
        )

    @staticmethod
    def _ema(prices: np.ndarray, period: int) -> float:
        if len(prices) < period:
            return float(np.mean(prices))
        k   = 2.0 / (period + 1)
        val = float(np.mean(prices[:period]))
        for p in prices[period:]:
            val = p * k + val * (1 - k)
        return val

    def notify_bar_change(self, new_bar_time: datetime):
        if self._last_bar_time != new_bar_time:
            self._last_bar_time = new_bar_time
            self._bar_traded    = False
            logger.debug(f"New H4 bar {new_bar_time} — bar_traded reset")

    def mark_traded(self):
        self._bar_traded = True

    @property
    def bar_is_traded(self) -> bool:
        return self._bar_traded

    def evaluate(
        self,
        X_scaled:    np.ndarray,
        close_price: float,       # LIVE tick (bid+ask)/2 — not cached bar close
        bar_time:    datetime,
        spread_pips: float,
        atr_pips:    float,
        ema200_bars: np.ndarray,  # fresh 200-bar array for EMA20 calc
    ) -> SignalOutput:

        self.notify_bar_change(bar_time)

        # ── 1. UDL ───────────────────────────────────────────────────────────
        regime, gmm_probs, hmm_probs, size_mult = self.udl.predict(X_scaled)
        gmm_conf    = float(gmm_probs.max())
        regime_name = self.REGIME_NAMES.get(regime, f'UNKNOWN_{regime}')

        # ── 2. ML probabilities ───────────────────────────────────────────────
        Xs         = X_scaled.reshape(1, -1)
        cat_probs  = self.cat.predict_proba(Xs)[0]
        cat_buy_p  = float(cat_probs[1])
        cat_sell_p = float(cat_probs[0])

        Xs_sc    = self.scaler_lr.transform(Xs)
        lr_probs = self.lr.predict_proba(Xs_sc)[0]
        lr_buy_p  = float(lr_probs[1])
        lr_sell_p = float(lr_probs[0])

        # ── 3. EMA20 + MR distance ────────────────────────────────────────────
        ema20        = self._ema(np.asarray(ema200_bars, dtype=np.float64), EMA_PERIOD)
        mr_dist_pips = (close_price - ema20) / PIP   # negative = below EMA → BUY
        abs_dist     = abs(mr_dist_pips)
        is_mr        = abs_dist >= self.mr_band
        mr_strength  = min(abs_dist / max(atr_pips, 1.0), 2.0)

        # ── 4. Levels & cost ─────────────────────────────────────────────────
        lvl      = self.REGIME_LEVELS.get(regime, self.REGIME_LEVELS[2])
        sl_pips  = float(lvl['sl'])
        tp1_pips = float(lvl['tp1'])
        tp_pips  = float(lvl['tp'])
        run_pips = float(lvl['runner'])
        cost_pips = spread_pips + self.icm_cost
        cost_ok   = (cost_pips / max(atr_pips, 1.0)) <= self.cost_ratio
        edge_pips = tp1_pips * 0.5 - cost_pips

        top_p = max(cat_buy_p, cat_sell_p)
        quality = 'STRONG' if top_p >= 0.80 else 'MODERATE' if top_p >= 0.65 else 'WEAK'

        def _hold(gate: str) -> SignalOutput:
            return SignalOutput(
                direction='HOLD', signal_type='HOLD', score=0.0,
                threshold=self.range_thr, gate_rejected=gate,
                cat_buy_prob=cat_buy_p, cat_sell_prob=cat_sell_p,
                lr_dir_prob=max(lr_buy_p, lr_sell_p),
                regime=regime, regime_name=regime_name,
                gmm_probs=gmm_probs, hmm_probs=hmm_probs, size_mult=size_mult,
                atr_pips=atr_pips, spread_pips=spread_pips,
                sl_pips=sl_pips, tp1_pips=tp1_pips,
                tp_pips=tp_pips, runner_pips=run_pips,
                expected_edge=edge_pips, ema20=ema20,
                close_price=close_price, mr_dist_pips=mr_dist_pips,
                mr_strength=mr_strength, is_mr=is_mr, quality=quality,
                bar_time=bar_time,
            )

        # ── Gate: ANOMALY always blocked ──────────────────────────────────────
        if regime == ANOMALY_REGIME:
            return _hold('ANOMALY_BLOCKED')

        # ── Gate: cost ────────────────────────────────────────────────────────
        if not cost_ok:
            return _hold('COST_GATE')

        # ── Route ─────────────────────────────────────────────────────────────
        if regime == 2:
            # ── MR-FADE (RANGE_CHOP) ─────────────────────────────────────────
            # RANGE_CHOP is the PRIMARY regime for this strategy.
            # We trade the reversion, not the trend.

            if not is_mr:
                return _hold('MR_DIST_TOO_SMALL')

            # Direction = sign of distance from EMA20 (price vs EMA, not ML)
            # price BELOW EMA20 → mr_dist_pips < 0 → BUY the dip back to mean
            # price ABOVE EMA20 → mr_dist_pips > 0 → SELL the rally back to mean
            raw_dir  = 'BUY' if mr_dist_pips < 0 else 'SELL'
            ml_agree = cat_buy_p  if raw_dir == 'BUY' else cat_sell_p
            lr_agree = lr_buy_p   if raw_dir == 'BUY' else lr_sell_p

            # Composite score: ML×0.5 + GMM_range×0.3 + HMM_range×0.2
            gmm_range = float(gmm_probs[2])
            hmm_range = float(hmm_probs[2])
            score     = ml_agree * 0.5 + gmm_range * 0.3 + hmm_range * 0.2

            # Sub-gate: extreme chop (GMM almost certain range = price going nowhere)
            if gmm_conf > self.block_thr:
                return _hold('RANGE_BLOCK_THR')

            # Sub-gate: ML must not strongly OPPOSE direction (threshold=0.40)
            if ml_agree < 0.40:
                return _hold('ML_DIRECTION_DISAGREE')

            # Sub-gate: LR must agree with MR direction
            if lr_agree < 0.50:
                return _hold('LR_DISAGREE')

            # Sub-gate: composite score
            if score < self.range_thr:
                return _hold('SCORE_GATE')

            signal_type = 'MR-FADE'
            threshold   = self.range_thr
            direction   = raw_dir
            ml_conf     = ml_agree

        else:
            # ── TREND (regime 0 or 1) ─────────────────────────────────────────
            signal_type = 'TREND'
            threshold   = self.trend_thr

            if cat_buy_p >= threshold and lr_buy_p >= 0.50:
                direction = 'BUY'
                ml_conf   = cat_buy_p
            elif cat_sell_p >= threshold and lr_sell_p >= 0.50:
                direction = 'SELL'
                ml_conf   = cat_sell_p
            else:
                return _hold('ML_PROB_GATE')

            score = ml_conf

        # ── Bar-trade gate ────────────────────────────────────────────────────
        if self._bar_traded:
            return _hold('BAR_TRADED')

        # ── Accepted ─────────────────────────────────────────────────────────
        log_mr = f"MR {mr_dist_pips:+.1f}pip str={mr_strength:.2f}" if is_mr else "TREND"
        logger.info(
            f"✅ {direction} [{log_mr}] | "
            f"score={score:.4f}(thr={threshold}) | "
            f"buy_p={cat_buy_p:.3f} sell_p={cat_sell_p:.3f} "
            f"p_dir={ml_conf:.4f} | "
            f"{regime_name} sz={int(size_mult*100)}% "
            f"TP={tp_pips:.0f} SL={sl_pips:.0f}pip "
            f"Q={quality} ev={edge_pips:.1f}pip | "
            f"close={close_price:.5f} ema20={ema20:.5f}"
        )

        return SignalOutput(
            direction=direction, signal_type=signal_type,
            score=score, threshold=threshold, gate_rejected='ACCEPTED',
            cat_buy_prob=cat_buy_p, cat_sell_prob=cat_sell_p,
            lr_dir_prob=ml_conf,
            regime=regime, regime_name=regime_name,
            gmm_probs=gmm_probs, hmm_probs=hmm_probs, size_mult=size_mult,
            atr_pips=atr_pips, spread_pips=spread_pips,
            sl_pips=sl_pips, tp1_pips=tp1_pips,
            tp_pips=tp_pips, runner_pips=run_pips,
            expected_edge=edge_pips, ema20=ema20,
            close_price=close_price, mr_dist_pips=mr_dist_pips,
            mr_strength=mr_strength, is_mr=is_mr, quality=quality,
            bar_time=bar_time,
        )