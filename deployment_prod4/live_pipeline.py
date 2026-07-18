"""
live_pipeline.py — v18.16  FIXED
==================================
Bug-fixes vs v18.15:
  BUG-1  Price / EMA frozen (same value every inference)
         Root cause: copy_rates_from_pos() was called once at bar-open
         and result cached.  Now LIVE tick price fetched every 10 s;
         EMA recalculated from fresh copy_rates every inference.

  BUG-2  SELL never executes
         Root cause: signal_engine was comparing buy_p > sell_p to pick
         direction, so price ABOVE EMA20 still generated BUY.
         Fixed in signal_engine.py (MR direction from price vs EMA20 sign).

  BUG-3  No second trade after TIME_EXIT
         Root cause: bar_traded flag never cleared on bar boundary.
         Fixed: signal_engine.notify_bar_change() called with current bar_time
         every inference; pipeline checks signal_engine.bar_is_traded.

  BUG-4  Virtual P&L stuck at -$0.22 in dashboard
         Root cause: state JSON written only on trade open/close, so dashboard
         saw stale data. Now state JSON written every inference.

  BUG-5  Virtual balance resets to $200 on every restart
         Root cause: balance seeded from MT5 real balance, not persisted.
         Fixed: balance persisted to state JSON and reloaded on startup.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

STATE_FILE  = Path(__file__).parent / "pipeline_state.json"
SYMBOL      = "EURUSD"
H4_SECONDS  = 4 * 3600
BARS_NEEDED = 220          # enough for EMA20 warm-up
LOOP_SEC    = 10           # inference every 10 seconds


# ──────────────────────────────────────────────────────────────────────────────
# Helper: state persistence
# ──────────────────────────────────────────────────────────────────────────────

def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_state(state: dict):
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
    except Exception as e:
        logger.warning(f"State save failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Feature builder (must match training exactly)
# ──────────────────────────────────────────────────────────────────────────────

def _build_X_scaled(broker, scaler_inp) -> Optional[np.ndarray]:
    """
    Pull last BARS_NEEDED H4 bars from MT5, compute 103 features, scale.
    Returns (1, 103) float64 array or None on error.
    """
    import MetaTrader5 as mt5
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H4, 0, BARS_NEEDED)
    if rates is None or len(rates) < 50:
        logger.warning("copy_rates returned None or too few bars")
        return None

    closes  = np.array([r['close'] for r in rates], dtype=np.float64)
    highs   = np.array([r['high']  for r in rates], dtype=np.float64)
    lows    = np.array([r['low']   for r in rates], dtype=np.float64)
    volumes = np.array([r['tick_volume'] for r in rates], dtype=np.float64)

    # ── 103-feature vector (same as training notebook) ────────────────────
    # This is a skeleton — replace with your real feature function if you have one.
    # The key requirement is that the output is (1,103) and uses the SAME
    # transformations as the training scaler.
    feats = _compute_103_features(closes, highs, lows, volumes)
    if feats is None:
        return None

    X = np.array(feats, dtype=np.float64).reshape(1, -1)
    if X.shape[1] != 103:
        logger.error(f"Feature count mismatch: got {X.shape[1]}, expected 103")
        return None
    if np.isnan(X).any() or np.isinf(X).any():
        logger.warning("NaN/Inf in features — skipping bar")
        return None

    return scaler_inp.transform(X)


def _compute_103_features(closes, highs, lows, volumes):
    """
    Compute the 103 base features used in training.
    IMPORTANT: this must match Cell 2 / feature engineering in NB-E-ML exactly.
    Replace this stub with your real implementation (imported from feature_builder.py).
    """
    try:
        from feature_builder import compute_features
        return compute_features(closes, highs, lows, volumes)
    except ImportError:
        # Fallback stub — DO NOT use in production
        logger.error("feature_builder.py not found — using zero stub (WRONG FEATURES)")
        return [0.0] * 103


# ──────────────────────────────────────────────────────────────────────────────
# Live bar helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_current_bar_time() -> Optional[datetime]:
    """Return the open-time of the current H4 bar (UTC)."""
    import MetaTrader5 as mt5
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H4, 0, 1)
    if rates is None or len(rates) == 0:
        return None
    return datetime.fromtimestamp(rates[0]['time'], tz=timezone.utc)


def _get_live_price() -> Optional[float]:
    """FIX BUG-1: fetch LIVE tick price, not cached bar close."""
    import MetaTrader5 as mt5
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return None
    return float((tick.bid + tick.ask) / 2.0)


def _get_spread_pips() -> float:
    import MetaTrader5 as mt5
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return 1.5
    return round((tick.ask - tick.bid) / 0.0001, 2)


def _get_atr_pips() -> float:
    import MetaTrader5 as mt5
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H4, 0, 20)
    if rates is None or len(rates) < 5:
        return 18.0
    trs = []
    for i in range(1, len(rates)):
        hi = rates[i]['high']; lo = rates[i]['low']; pc = rates[i-1]['close']
        trs.append(max(hi - lo, abs(hi - pc), abs(lo - pc)))
    return round(np.mean(trs[-14:]) / 0.0001, 1)


def _get_ema200_bars() -> Optional[np.ndarray]:
    """FIX BUG-1: return fresh 220-bar close array for EMA recalculation."""
    import MetaTrader5 as mt5
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H4, 0, BARS_NEEDED)
    if rates is None or len(rates) < 30:
        return None
    return np.array([r['close'] for r in rates], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# LivePipeline
# ──────────────────────────────────────────────────────────────────────────────

class LivePipeline:

    def __init__(self, signal_engine, trade_manager, broker, scaler_inp,
                 cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.sig   = signal_engine
        self.tm    = trade_manager
        self.brok  = broker
        self.scaler_inp = scaler_inp
        self.mode  = cfg.get('mode', 'PAPER')
        self.pip_value = cfg.get('pip_value', 10.0)

        # ── FIX BUG-5: load persisted virtual balance ─────────────────────
        saved = _load_state()
        mt5_bal = self.brok.get_balance() or 200.0
        if 'virtual_balance' in saved:
            self.virtual_balance = float(saved['virtual_balance'])
            logger.info(f"Virtual balance restored: ${self.virtual_balance:.2f}")
        else:
            self.virtual_balance = mt5_bal
            logger.info(f"Virtual balance seeded: ${self.virtual_balance:.2f}")

        logger.info(
            f"[{self.mode}] v18.16 | {SYMBOL} H4 | EMA20-MR SELL/BUY | "
            f"virtual balance | gates every inference | "
            f"risk=1% daily_stop=3%"
        )
        logger.info(f"MT5=${mt5_bal:.2f} | Virtual=${self.virtual_balance:.2f}")

        self._inference_count = 0
        self._last_bar_time: Optional[datetime] = None

    # ── State snapshot (written every inference) ──────────────────────────
    def _write_state(self, sig_out=None):
        """FIX BUG-4: write state every loop so dashboard always sees fresh data."""
        state = {
            'ts':              datetime.now(timezone.utc).isoformat(),
            'virtual_balance': round(self.virtual_balance, 2),
            'mt5_balance':     self.brok.get_balance(),
            'inference_count': self._inference_count,
            'open_trades':     len(self.tm.open_trades),
            'closed_trades':   len(self.tm.closed_trades),
            'summary':         self.tm.summary(),
        }
        if sig_out is not None:
            state['last_signal'] = {
                'direction':   sig_out.direction,
                'score':       round(sig_out.score, 4),
                'gate':        sig_out.gate_rejected,
                'ema20':       round(sig_out.ema20, 5),
                'close':       round(sig_out.close_price, 5),
                'mr_dist':     round(sig_out.mr_dist_pips, 1),
                'buy_p':       round(sig_out.cat_buy_prob, 4),
                'sell_p':      round(sig_out.cat_sell_prob, 4),
                'regime':      sig_out.regime_name,
                'atr':         round(sig_out.atr_pips, 1),
                'spread':      round(sig_out.spread_pips, 2),
                'gmm_probs':   [round(float(p), 4) for p in sig_out.gmm_probs],
                'hmm_probs':   [round(float(p), 4) for p in sig_out.hmm_probs],
                'bar_time':    sig_out.bar_time.isoformat() if sig_out.bar_time else None,
            }
        _save_state(state)

    # ── Single inference cycle ────────────────────────────────────────────
    def _inference_cycle(self):
        self._inference_count += 1

        # --- 1. Fresh market data (FIX BUG-1) --------------------------------
        close_price = _get_live_price()
        if close_price is None:
            logger.warning("No live price — skipping cycle")
            self._write_state()
            return

        spread_pips  = _get_spread_pips()
        atr_pips     = _get_atr_pips()
        ema200_bars  = _get_ema200_bars()
        if ema200_bars is None:
            logger.warning("No bar data — skipping cycle")
            self._write_state()
            return

        bar_time = _get_current_bar_time()
        if bar_time is None:
            logger.warning("Cannot determine current bar time")
            self._write_state()
            return

        # FIX BUG-3/4: notify signal engine of bar change (resets bar_traded)
        self.sig.notify_bar_change(bar_time)

        # --- 2. Feature vector ------------------------------------------------
        X_scaled = _build_X_scaled(self.brok, self.scaler_inp)
        if X_scaled is None:
            self._write_state()
            return

        # --- 3. Signal --------------------------------------------------------
        sig = self.sig.evaluate(
            X_scaled    = X_scaled,
            close_price = close_price,       # LIVE tick, not cached bar close
            bar_time    = bar_time,
            spread_pips = spread_pips,
            atr_pips    = atr_pips,
            ema200_bars = ema200_bars,       # fresh bars for EMA recalculation
        )

        # --- 4. Update open trades --------------------------------------------
        mt5_balance = self.brok.get_balance() or self.virtual_balance
        trades_to_close = []
        for tid, t in list(self.tm.open_trades.items()):
            result = self.tm.update(tid, close_price, mt5_balance)
            action = result.get('action', 'none')
            if action in ('close_all', 'close_tp1'):
                pnl_pips = result.get('pnl_pips', 0.0)
                if action == 'close_all':
                    self.virtual_balance += pnl_pips * t.lot_size * self.pip_value * 10
                    reason = result.get('reason', '?')
                    logger.info(
                        f"[{self.mode}] close_all: pnl={pnl_pips:.1f}pip reason={reason}"
                    )
                    logger.info(
                        f"[{self.mode}] {pnl_pips:.1f}pip × {t.lot_size:.2f}lots "
                        f"= ${pnl_pips * t.lot_size * self.pip_value * 10:.2f} "
                        f"→ virtual=${self.virtual_balance:.2f}"
                    )
                elif action == 'close_tp1':
                    self.virtual_balance += pnl_pips * t.tp1_lots * self.pip_value * 10
                    logger.info(f"[{self.mode}] TP1 partial close: pnl={pnl_pips:.1f}pip")

        # --- 5. Open new trade if signal accepted ----------------------------
        if (sig.direction != 'HOLD'
                and sig.gate_rejected == 'ACCEPTED'
                and not self.sig.bar_is_traded):

            trade = self.tm.open_trade(
                signal      = sig,
                entry_price = close_price,
                balance     = mt5_balance,
                symbol      = SYMBOL,
            )
            if trade is not None:
                self.sig.mark_traded()
                logger.info(
                    f"[{self.mode}] OPEN {sig.direction} "
                    f"entry={close_price:.5f} "
                    f"sl={trade.sl_price:.5f} "
                    f"tp1={trade.tp1_price:.5f} "
                    f"lots={trade.lot_size:.2f} "
                    f"score={sig.score:.2f} mr={sig.is_mr}"
                )

        # --- 6. FIX BUG-4: always write state --------------------------------
        self._write_state(sig)

    # ── Main run loop ─────────────────────────────────────────────────────
    def run(self):
        logger.info(f"[{self.mode}] LivePipeline started — loop every {LOOP_SEC}s")
        while True:
            try:
                if not self.brok.is_connected():
                    logger.warning("Broker disconnected — reconnecting...")
                    self.brok.reconnect()

                self._inference_cycle()

            except Exception as e:
                logger.exception(f"Inference cycle error: {e}")

            time.sleep(LOOP_SEC)


# ──────────────────────────────────────────────────────────────────────────────
# Entry-point helper (called from demo_trading.py / START_ALL.bat)
# ──────────────────────────────────────────────────────────────────────────────

def build_and_run(cfg_path: str = "config.json"):
    import pickle, sys
    from pathlib import Path

    # Load config
    cfg_file = Path(cfg_path)
    cfg = json.loads(cfg_file.read_text()) if cfg_file.exists() else {}

    base_dir  = Path(__file__).parent
    udl_path  = cfg.get('udl_path',  str(base_dir / 'udl_engine_v3.pkl'))
    ml_path   = cfg.get('ml_path',   str(base_dir / 'ml_models.pkl'))

    # Load models
    from udl_engine import UDLEngine
    udl = UDLEngine(udl_path)

    with open(ml_path, 'rb') as f:
        ml = pickle.load(f)
    cat_clf    = ml['cat_clf']
    lr_clf     = ml['lr_clf']
    scaler_lr  = ml['scaler_lr']
    scaler_inp = ml['scaler_inp']     # 103-feature input scaler

    from signal_engine import SignalEngine
    sig_eng = SignalEngine(udl, cat_clf, lr_clf, scaler_lr, cfg)

    from trade_manager import TradeManager
    tm = TradeManager(cfg)

    from icm_broker import ICMBroker
    broker = ICMBroker(cfg)
    broker.connect()

    pipeline = LivePipeline(sig_eng, tm, broker, scaler_inp, cfg)
    pipeline.run()


if __name__ == '__main__':
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler('v1816.log', encoding='utf-8'),
            logging.StreamHandler(),
        ]
    )
    build_and_run(cfg)