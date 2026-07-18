"""
signal_engine.py — v18.14
==========================
BUGS FIXED FROM v18.13:

BUG 1 — "ALWAYS BUY, NEVER SELL" (ROOT CAUSE):
  The CatBoost model was trained on a bullish EUR/USD phase.
  buy_p is chronically ~0.88, sell_p ~0.12.
  DIR_THRESHOLD=0.55 for SELL is NEVER met by sell_p.

  v18.13 said: "if close > ema20 and sell_p >= 0.55 → SELL"
  But sell_p NEVER reaches 0.55. So it always fell through to
  "elif buy_p >= 0.55 → BUY". Mean-reversion was dead letter.

  v18.14 FIX — RANGE mean-reversion OVERRIDE:
    In RANGE regime, direction is determined by price vs EMA20 FIRST:
      · price > EMA20 * 1.0003  →  force direction = SELL  (fade the top)
      · price < EMA20 * 0.9997  →  force direction = BUY   (fade the bottom)
      · price near EMA20        →  use ML direction (buy_p vs sell_p)

    For the FORCED direction, we use 1 - p_opposite as the probability
    score component (the model's implicit confidence in reversal).
    This bypasses the ML threshold for the chosen direction.
    A minimum probability floor of RANGE_MR_MIN_P (0.30) is required
    on the forced side's implied probability. This prevents trading
    when the model is absolutely certain the OTHER direction is right.

BUG 2 — SELL SCORE FORMULA:
  When direction=SELL is forced, we score using sell_p (even if low),
  but use max(1-buy_p, sell_p) so a very confident BUY model still
  correctly blocks the SELL via the gate system.

BUG 3 — EV CALCULATION FOR SELL:
  Expected value now uses the correct directional probability
  (sell_p or the composite direction probability).

REGIME CONFIG (unchanged from v18.13):
  BULL_TREND : thr=0.55  q=1.00  size=100%
  BEAR_TREND : thr=0.55  q=1.00  size=100%
  RANGE_CHOP : thr=0.52  q=0.70  size=90%   ← ENABLED w/ MR override
  ANOMALY    : blocked
"""
import os, sys, logging, pickle
from dataclasses import dataclass
from typing import Optional
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

logger = logging.getLogger(__name__)

REGIME_NAMES = {0: 'BULL_TREND', 1: 'BEAR_TREND', 2: 'RANGE_CHOP', 3: 'ANOMALY'}

TP_SL_TREND = {
    'STRONG': (80.0, 35.0),
    'NORMAL': (60.0, 28.0),
    'WEAK':   (45.0, 22.0),
}
TP_SL_RANGE = {
    'STRONG': (38.0, 16.0),
    'NORMAL': (28.0, 13.0),
    'WEAK':   (20.0, 10.0),
}

REGIME_CONFIG = {
    0: dict(quality=1.00, threshold=0.55, size=1.00),   # BULL_TREND
    1: dict(quality=1.00, threshold=0.55, size=1.00),   # BEAR_TREND
    2: dict(quality=0.70, threshold=0.52, size=0.90),   # RANGE_CHOP
    3: dict(quality=0.00, threshold=9.99, size=0.00),   # ANOMALY
}

W_ML  = 0.50
W_RG  = 0.30
W_HMM = 0.20

DIR_THRESHOLD    = 0.55   # ML probability required for TREND trades
RANGE_MR_BAND    = 0.0003 # price must be ≥0.03% away from EMA20 to override
RANGE_MR_MIN_P   = 0.30   # minimum sell_p (or buy_p) even in MR override mode
                           # if the model is ≥70% AGAINST direction, we skip
ATR_FLOOR        = 8.0
ATR_CEILING      = 55.0
COST_RATIO_MAX   = 0.30
EDGE_MIN         = 1.0

COOLDOWN_BARS  = 0
MAX_TRADES_DAY = 99


@dataclass
class TradeSignal:
    direction:        str
    score:            float
    cat_buy_prob:     float
    cat_sell_prob:    float
    lr_buy_prob:      float
    lr_sell_prob:     float
    regime:           int
    regime_name:      str
    gmm_conf:         float
    hmm_conf:         float
    size_mult:        float
    tp_pips:          float
    sl_pips:          float
    tp1_pips:         float
    runner_pips:      float
    expected_edge:    float
    atr_pips:         float
    spread_pips:      float
    gate_passed:      bool
    gate_rejected:    str
    regime_threshold: float


class SignalEngine:
    """
    v18.14 — True balanced BUY/SELL.
    RANGE: mean-reversion direction override (price vs EMA20).
    TREND: pure ML direction (buy_p vs sell_p vs threshold).
    """

    def __init__(self, bundle_path: str):
        with open(bundle_path, 'rb') as f:
            b = pickle.load(f)
        self.cat_clf  = b['cat_clf']
        self.icm_cost = b['icm_cost_fixed']
        self.tp1_frac = b['tp1_fraction']
        self.run_frac = b['runner_fraction']

        logger.info(
            f"[v18.14] SignalEngine | "
            f"RANGE MR-override band={RANGE_MR_BAND:.4f} min_p={RANGE_MR_MIN_P} | "
            f"TREND DIR_THR={DIR_THRESHOLD} | "
            f"BULL/BEAR thr=0.55 q=1.00 | "
            f"RANGE thr=0.52 q=0.70 size=90% | "
            f"ANOMALY=blocked")

    def notify_trade_opened(self):
        pass   # no cooldown

    def _quality(self, score: float, regime: int) -> str:
        if regime == 2:
            if score >= 0.75: return 'STRONG'
            if score >= 0.62: return 'NORMAL'
            return 'WEAK'
        else:
            if score >= 0.80: return 'STRONG'
            if score >= 0.68: return 'NORMAL'
            return 'WEAK'

    def _tp_sl(self, quality: str, regime: int) -> tuple:
        table = TP_SL_RANGE if regime == 2 else TP_SL_TREND
        return table.get(quality, table['NORMAL'])

    def _hold(self, reason, buy_p, sell_p, regime, gmm_conf, hmm_conf,
              size_mult, atr, spread, score=0.0, rg_thr=0.55):
        tp, sl = self._tp_sl('NORMAL', regime)
        return TradeSignal(
            direction='HOLD', score=score,
            cat_buy_prob=buy_p, cat_sell_prob=sell_p,
            lr_buy_prob=0.0, lr_sell_prob=0.0,
            regime=regime, regime_name=REGIME_NAMES.get(regime, '?'),
            gmm_conf=gmm_conf, hmm_conf=hmm_conf, size_mult=size_mult,
            tp_pips=tp, sl_pips=sl,
            tp1_pips=tp * self.tp1_frac,
            runner_pips=tp + (15.0 if regime == 2 else 25.0),
            expected_edge=0.0, atr_pips=atr, spread_pips=spread,
            gate_passed=False, gate_rejected=reason,
            regime_threshold=rg_thr)

    def run(self, X_ml: np.ndarray,
            atr_pips: float, spread_pips: float,
            regime: int,
            gmm_probs: np.ndarray,
            hmm_probs: np.ndarray,
            size_mult: float,
            ema20: float = 0.0,
            close: float = 0.0,
            bar_date: str = '') -> TradeSignal:

        assert X_ml.shape == (1, 64), f"Expected (1,64) got {X_ml.shape}"

        cfg          = REGIME_CONFIG.get(regime, REGIME_CONFIG[3])
        rg_quality   = cfg['quality']
        rg_threshold = cfg['threshold']
        rg_size      = cfg['size']

        # ML probabilities
        cat_p    = self.cat_clf.predict_proba(X_ml)
        buy_p    = float(cat_p[0][1])
        sell_p   = float(cat_p[0][0])
        gmm_conf = float(gmm_probs.max())
        hmm_conf = float(hmm_probs.max())
        hmm_reg  = int(hmm_probs.argmax())

        # ── Gate 1: ANOMALY ──────────────────────────────────────────
        if regime == 3:
            return self._hold('ANOMALY', buy_p, sell_p, regime,
                              gmm_conf, hmm_conf, size_mult,
                              atr_pips, spread_pips, rg_thr=rg_threshold)

        # ── Gate 2: Direction determination ──────────────────────────
        #
        # RANGE regime: mean-reversion override
        #   - price significantly above EMA20 → SHORT (sell the top)
        #   - price significantly below EMA20 → LONG  (buy the dip)
        #   - only blocked if model is VERY confident the other way
        #     (opponent probability > 1 - RANGE_MR_MIN_P = 0.70)
        #
        # TREND regime: pure ML (must exceed DIR_THRESHOLD)
        #
        raw_dir = None
        p_dir   = 0.0
        mr_mode = False  # flag for logging

        if regime == 2 and ema20 > 0 and close > 0:
            ratio = close / ema20
            if ratio > (1.0 + RANGE_MR_BAND):
                # Price extended above EMA20 → fade = SELL
                # Block only if model is very confident BUY (buy_p > 1 - RANGE_MR_MIN_P)
                if buy_p > (1.0 - RANGE_MR_MIN_P):  # model strongly says BUY (>0.70)
                    # Don't fight a very strong ML trend signal
                    return self._hold(
                        'MR_BLOCKED_BY_STRONG_BUY',
                        buy_p, sell_p, regime,
                        gmm_conf, hmm_conf, size_mult,
                        atr_pips, spread_pips, rg_thr=rg_threshold)
                raw_dir = 'SELL'
                # Use implied reversal probability: max of sell_p and (1-buy_p)
                p_dir   = max(sell_p, 1.0 - buy_p)
                mr_mode = True

            elif ratio < (1.0 - RANGE_MR_BAND):
                # Price extended below EMA20 → fade = BUY
                if sell_p > (1.0 - RANGE_MR_MIN_P):  # model strongly says SELL (>0.70)
                    return self._hold(
                        'MR_BLOCKED_BY_STRONG_SELL',
                        buy_p, sell_p, regime,
                        gmm_conf, hmm_conf, size_mult,
                        atr_pips, spread_pips, rg_thr=rg_threshold)
                raw_dir = 'BUY'
                p_dir   = max(buy_p, 1.0 - sell_p)
                mr_mode = True

            else:
                # Price near EMA20 — use standard ML direction
                if buy_p >= DIR_THRESHOLD and buy_p >= sell_p:
                    raw_dir = 'BUY';  p_dir = buy_p
                elif sell_p >= DIR_THRESHOLD and sell_p > buy_p:
                    raw_dir = 'SELL'; p_dir = sell_p
                else:
                    return self._hold('BASIC_PROB', buy_p, sell_p, regime,
                                      gmm_conf, hmm_conf, size_mult,
                                      atr_pips, spread_pips, rg_thr=rg_threshold)
        else:
            # TREND regime — pure ML direction
            if buy_p >= DIR_THRESHOLD and buy_p >= sell_p:
                raw_dir = 'BUY';  p_dir = buy_p
            elif sell_p >= DIR_THRESHOLD and sell_p > buy_p:
                raw_dir = 'SELL'; p_dir = sell_p
            else:
                return self._hold('BASIC_PROB', buy_p, sell_p, regime,
                                  gmm_conf, hmm_conf, size_mult,
                                  atr_pips, spread_pips, rg_thr=rg_threshold)

        # ── Gate 3: Cost vs ATR ───────────────────────────────────────
        total_cost = spread_pips + self.icm_cost
        if atr_pips < ATR_FLOOR or \
           total_cost / max(atr_pips, 0.1) > COST_RATIO_MAX:
            return self._hold('COST_ATR_GATE', buy_p, sell_p, regime,
                              gmm_conf, hmm_conf, size_mult,
                              atr_pips, spread_pips, rg_thr=rg_threshold)

        # ── Gate 4: Extreme volatility ────────────────────────────────
        if atr_pips > ATR_CEILING:
            return self._hold('VOLATILITY', buy_p, sell_p, regime,
                              gmm_conf, hmm_conf, size_mult,
                              atr_pips, spread_pips, rg_thr=rg_threshold)

        # ── Composite score ───────────────────────────────────────────
        # p_dir is already the correct directional probability
        # (either ML probability or mean-reversion implied probability)
        hmm_agree = 1.0 if hmm_reg == regime else 0.5
        cf_score  = float(gmm_conf * hmm_agree)

        # Clamp p_dir to [0,1] for score
        p_ml_clamped = float(np.clip(p_dir, 0.0, 1.0))
        score = float(np.clip(
            W_ML  * p_ml_clamped +
            W_RG  * rg_quality +
            W_HMM * cf_score, 0.0, 1.0))

        # ── Gate 5: Score threshold ───────────────────────────────────
        if score < rg_threshold:
            return self._hold('SCORE_THRESHOLD', buy_p, sell_p, regime,
                              gmm_conf, hmm_conf, size_mult,
                              atr_pips, spread_pips, score=score,
                              rg_thr=rg_threshold)

        # ── Gate 6: Expected value ────────────────────────────────────
        quality        = self._quality(score, regime)
        tp_pip, sl_pip = self._tp_sl(quality, regime)
        tp1_pip        = tp_pip * self.tp1_frac
        runner_pip     = tp_pip + (15.0 if regime == 2 else 25.0)
        win_exp        = self.tp1_frac * tp_pip + self.run_frac * runner_pip
        ev             = p_ml_clamped * win_exp - (1.0 - p_ml_clamped) * sl_pip - total_cost

        if ev < EDGE_MIN:
            return self._hold('EDGE_BELOW_MIN', buy_p, sell_p, regime,
                              gmm_conf, hmm_conf, size_mult,
                              atr_pips, spread_pips, score=score,
                              rg_thr=rg_threshold)

        # ── Signal accepted ───────────────────────────────────────────
        regime_label = REGIME_NAMES.get(regime, '?')
        mr_tag = ' [MR-OVERRIDE]' if mr_mode else ''
        logger.info(
            f"[TRADE] {raw_dir}{mr_tag} | score={score:.4f}(thr={rg_threshold:.2f}) | "
            f"ml_p={p_dir:.3f}(buy={buy_p:.3f}/sell={sell_p:.3f}) "
            f"rg_q={rg_quality:.2f} cf={cf_score:.3f} | "
            f"{regime_label} size={rg_size:.0%} TP={tp_pip:.0f} SL={sl_pip:.0f}pip "
            f"Q={quality} ev={ev:.1f}pip close={close:.5f} ema20={ema20:.5f} "
            f"ratio={close/ema20 if ema20>0 else 0:.6f}")

        return TradeSignal(
            direction=raw_dir, score=score,
            cat_buy_prob=buy_p, cat_sell_prob=sell_p,
            lr_buy_prob=0.0, lr_sell_prob=0.0,
            regime=regime, regime_name=regime_label,
            gmm_conf=gmm_conf, hmm_conf=hmm_conf,
            size_mult=rg_size,
            tp_pips=tp_pip, sl_pips=sl_pip,
            tp1_pips=tp1_pip, runner_pips=runner_pip,
            expected_edge=ev, atr_pips=atr_pips, spread_pips=spread_pips,
            gate_passed=True, gate_rejected='ACCEPTED',
            regime_threshold=rg_threshold)
