"""
live_pipeline.py — v18.14
==========================
BUGS FIXED from v18.13:

BUG 1 — BALANCE STUCK AT $200:
  In paper trading mode, balance was ONLY updated by set_balance()
  which calls broker.get_balance() (real MT5 demo balance = always $200).
  Paper trades never affected the displayed balance.

  FIX: self.virtual_balance tracks paper P&L separately.
  Paper mode: balance shown = virtual_balance (starts at MT5 balance,
  updated by +/- pnl_pips * pip_value on each close).
  Live mode:  balance shown = broker.get_balance() as before.

BUG 2 — PnL ALWAYS 0.0 AFTER FIRST TRADE:
  The trade_manager.summary() total_pnl was correct in pips but
  the JSON field 'total_pnl' was never connected to virtual_balance.

  FIX: on every trade close action detected, update virtual_balance.

BUG 3 — UNREALIZED PnL NOT UPDATING:
  update_price_only() computed pnl_unrealised correctly but the
  bar-update JSON overwrite reset it to 0 on next ML cycle.

  FIX: on_bar_close() now computes live unrealised PnL from price.

BUG 4 — BALANCE IN JSON ALWAYS SHOWING MT5 BALANCE:
  FIX: JSON 'balance' field now shows virtual_balance in paper mode.
"""
import os, sys, pickle, logging, json, csv
import numpy as np, pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
os.chdir(_DIR)

from feature_builder import (build_raw_features, features_to_vector,
                              build_full_matrix, get_ema20_close)
from udl_engine    import UDLEngine
from signal_engine import SignalEngine
from trade_manager import TradeManager

_BUNDLE        = os.path.join(_DIR, 'deploy_ml_bundle_v18_7.pkl')
_B_SCALED      = os.path.join(_DIR, 'B_scaled_v6_clean.pkl')
_UDL           = os.path.join(_DIR, 'udl_engine_v3.pkl')
_JSON          = os.path.join(_DIR, 'live_data.json')
_SYMBOL        = 'EURUSD'
_RISK          = 0.01
_MAX_CONC      = 3
_DAILY_LIM     = 0.03
_SLIP_N        = 0.5
_SLIP_NEWS     = 2.0
_MAX_HOLD_BARS = 12
_PIP_VALUE     = 10.0   # USD per pip per 0.01 lot (standard lot=100k, 1pip=$10)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(_DIR, 'v1813.log'), encoding='utf-8')
    ])
logger = logging.getLogger('LivePipeline')

REGIME = {0: 'BULL_TREND', 1: 'BEAR_TREND', 2: 'RANGE_CHOP', 3: 'ANOMALY'}


def _check_files():
    missing = [os.path.basename(p)
               for p in [_BUNDLE, _B_SCALED, _UDL] if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"\n❌ Missing pkl files in {_DIR}:\n" +
            "\n".join(f"   {f}" for f in missing) +
            "\n\nCopy into deployment_prod folder:\n"
            "  deploy_ml_bundle_v18_7.pkl\n"
            "  B_scaled_v6_clean.pkl\n"
            "  udl_engine_v3.pkl")


def _atomic_json(data: dict):
    try:
        tmp = _JSON + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, default=str)
        if os.path.exists(_JSON):
            os.replace(tmp, _JSON)
        else:
            os.rename(tmp, _JSON)
    except Exception as e:
        logger.debug(f"JSON write skipped: {e}")


def _open_positions_list(mgr: TradeManager, price: float = 0.0) -> list:
    out = []
    for tid, t in mgr.open_trades.items():
        d_str = 'BUY' if t.direction == 1 else 'SELL'
        pnl_u = 0.0
        if price > 0:
            pnl_u = round(t.direction * (price - t.entry_price) / 0.0001, 1)
        out.append({
            'trade_id':      tid,
            'direction':     d_str,
            'entry':         round(t.entry_price, 5),
            'sl':            round(t.sl_price, 5),
            'tp1':           round(t.tp1_price, 5),
            'runner_tp':     round(t.runner_tp, 5),
            'lots':          round(t.lot_size, 2),
            'sl_pips':       round(t.sl_pips, 1),
            'tp_pips':       round(t.tp_pips, 1),
            'atr_pips':      round(t.atr_pips, 1),
            'bars_open':     t.bars_open,
            'tp1_closed':    t.tp1_closed,
            'score':         round(t.signal_prob, 4),
            'regime':        REGIME.get(t.regime, '?'),
            'entry_time':    str(t.entry_time),
            'pnl_realised':  round(t.realized_pnl, 2),
            'pnl_unrealised': pnl_u,
        })
    return out


class LivePipeline:

    def __init__(self, paper_trading: bool = True,
                 log_subdir: str = 'demo_logs'):
        _check_files()
        self.paper            = paper_trading
        self.symbol           = _SYMBOL
        self.balance          = 10_000.0   # MT5 / seed balance
        self.virtual_balance  = 10_000.0   # paper P&L tracker (paper mode only)
        self._vb_initialised  = False       # set to True on first set_balance()
        self.log_dir = Path(_DIR) / log_subdir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        with open(_B_SCALED, 'rb') as f:
            B = pickle.load(f)
        self.b_scaler = B['scaler']

        with open(_BUNDLE, 'rb') as f:
            bndl = pickle.load(f)
        self.top_k = np.array(bndl['top_k_idx'])

        self.udl    = UDLEngine(_UDL)
        self.engine = SignalEngine(_BUNDLE)
        self.mgr    = TradeManager({
            'max_hold_bars':        _MAX_HOLD_BARS,
            'tp1_fraction':         bndl['tp1_fraction'],
            'runner_fraction':      bndl['runner_fraction'],
            'icm_cost_fixed':       bndl['icm_cost_fixed'],
            'trail_atr_mult':       3.0,
            'max_concurrent':       _MAX_CONC,
            'risk_per_trade':       _RISK,
            'pip_value':            _PIP_VALUE,
            'pip_size':             0.0001,
            'slippage_pips_normal': _SLIP_N,
            'slippage_pips_news':   _SLIP_NEWS,
            'daily_loss_limit_pct': _DAILY_LIM,
        })

        self.bar_count        = 0
        self.sig_counts       = Counter()
        self.gate_counts      = Counter()
        self.inference_count  = 0
        self._last_sig        = None
        self._last_ml_ts      = None
        self._traded_this_bar = False
        self._last_price      = 0.0
        self._last_spread     = 1.5

        tag = '[PAPER]' if paper_trading else '[LIVE]'
        logger.info(
            f"{tag} v18.14 | {_SYMBOL} H4 | "
            f"RANGE MR-override | SELL signals enabled | "
            f"1trade/bar | direction lock | "
            f"risk={_RISK*100:.0f}% | daily_stop={_DAILY_LIM*100:.0f}% | "
            f"virtual_balance tracking={'ON (paper)' if paper_trading else 'OFF (live MT5)'}")

    def set_balance(self, b: float):
        """Called with real MT5 balance. Seeds virtual_balance on first call."""
        self.balance = b
        if not self._vb_initialised:
            self.virtual_balance = b
            self._vb_initialised = True
            logger.info(f"Balance seeded: ${b:,.2f} (virtual)")
        else:
            if not self.paper:
                # Live mode: track real balance
                self.virtual_balance = b
            # Paper mode: don't overwrite virtual_balance with MT5 balance
            logger.info(f"MT5 Balance: ${b:,.2f} | Virtual: ${self.virtual_balance:,.2f}")

    def _update_virtual_balance(self, pnl_pips: float, lots: float):
        """Update virtual balance after a paper trade closes."""
        if self.paper:
            # pnl_pips * pip_value_per_lot * lots
            # For EURUSD: 1 pip = $10 per standard lot (1.0 lot)
            # 0.01 lot = $0.10 per pip
            pnl_usd = pnl_pips * _PIP_VALUE * lots
            self.virtual_balance += pnl_usd
            logger.info(
                f"[PAPER] PnL: {pnl_pips:+.1f}pip × ${_PIP_VALUE:.0f} × {lots:.2f}lots "
                f"= ${pnl_usd:+.2f} | New virtual balance: ${self.virtual_balance:,.2f}")

    def _display_balance(self) -> float:
        """Return the balance to display/use for risk calculations."""
        if self.paper:
            return self.virtual_balance
        return self.balance

    def update_price_only(self, price: float, spread_pips: float) -> None:
        """1-second price tick — zero ML cost, updates unrealised PnL."""
        self._last_price  = price
        self._last_spread = spread_pips
        if self._last_sig is None:
            return
        positions = _open_positions_list(self.mgr, price)
        try:
            existing = {}
            if os.path.exists(_JSON):
                with open(_JSON, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
        except Exception:
            existing = {}
        existing.update({
            'price':          round(price, 5),
            'spread':         round(spread_pips, 2),
            'ts':             datetime.now(timezone.utc).isoformat(),
            'open_trades':    len(self.mgr.open_trades),
            'open_positions': positions,
            'balance':        round(self._display_balance(), 2),
        })
        _atomic_json(existing)

    def on_bar_update(self, df: pd.DataFrame, price: float,
                      spread_pips: float, is_new_bar: bool = False) -> dict:
        if is_new_bar:
            self._traded_this_bar = False
        return self.on_bar_close(
            df=df, price=price, spread_pips=spread_pips,
            is_news=False,
            _trade_allowed=is_new_bar,
            _write_signals=is_new_bar)

    def _direction_locked(self, sig_direction: str) -> bool:
        for trade in self.mgr.open_trades.values():
            if ('BUY' if trade.direction == 1 else 'SELL') == sig_direction:
                return True
        return False

    def _signal_exit_check(self, sig, price: float) -> list:
        to_close = []
        for tid, trade in list(self.mgr.open_trades.items()):
            if sig.regime == 3:
                to_close.append((tid, 'SIGNAL_ANOMALY'))
                continue
            if not sig.gate_passed:
                continue
            if trade.direction == 1 and sig.direction == 'SELL':
                to_close.append((tid, 'SIGNAL_FLIP'))
            elif trade.direction == -1 and sig.direction == 'BUY':
                to_close.append((tid, 'SIGNAL_FLIP'))
        return to_close

    def on_bar_close(self, df: pd.DataFrame, price: float,
                     spread_pips: float, is_news: bool = False,
                     _trade_allowed: bool = True,
                     _write_signals: bool = True) -> dict:
        if _trade_allowed:
            self.bar_count += 1
        self.inference_count += 1
        self._last_ml_ts  = datetime.now(timezone.utc).isoformat()
        self._last_price  = price
        self._last_spread = spread_pips

        ts = df.index[-1]
        try:
            bar_date = pd.Timestamp(ts).strftime('%Y-%m-%d')
        except Exception:
            bar_date = ''

        try:
            raw_f = build_raw_features(df)
            raw_v = features_to_vector(raw_f)
            ema20, close = get_ema20_close(df)
        except Exception as e:
            logger.error(f"Feature error at {ts}: {e}")
            return {'status': 'ERROR', 'reason': str(e)}

        atr  = float(raw_f.get('atr_14_pips', 25.0))
        X_sc = self.b_scaler.transform(raw_v.reshape(1, -1))

        regime, gp, hp, sm = self.udl.predict(X_sc)
        X_ml = build_full_matrix(raw_v, self.b_scaler, gp, hp, sm, self.top_k)

        sig = self.engine.run(
            X_ml, atr, spread_pips, regime, gp, hp, sm,
            ema20=ema20, close=close, bar_date=bar_date)

        self._last_sig = sig

        # ── Early exits (signal flip / anomaly) ──────────────────────
        actions = []
        for tid, reason in self._signal_exit_check(sig, price):
            if tid in self.mgr.open_trades:
                trade = self.mgr.open_trades[tid]
                pnl   = trade.direction * (price - trade.entry_price) / 0.0001
                pnl  -= trade.slippage_pips + 1.2
                pnl   = round(pnl, 1)
                logger.info(f"EARLY_EXIT({reason})|{tid}|pnl={pnl:.1f}pip")
                act = {'action': 'close_all', 'pnl_pips': pnl, 'reason': reason,
                       'lots': trade.lot_size}
                actions.append(act)
                self._log_close(trade, price, pnl, reason)
                # Update virtual balance
                self._update_virtual_balance(pnl, trade.lot_size)
                self.mgr._close(trade, price)

        # ── Normal trade manager updates ──────────────────────────────
        for tid in list(self.mgr.open_trades.keys()):
            trade_before = self.mgr.open_trades.get(tid)
            lots_before  = trade_before.lot_size if trade_before else 0.01
            res = self.mgr.update(tid, price, self._display_balance())
            if res['action'] != 'none':
                actions.append(res)
                self._log_action(res)
                # Update virtual balance on any close event
                if res['action'] in ('close_all',) and tid not in self.mgr.open_trades:
                    pnl_c = res.get('pnl_pips', 0.0)
                    self._update_virtual_balance(pnl_c, lots_before)
                    # Write CLOSE row to CSV
                    closed = next((c for c in self.mgr.closed_trades
                                   if c['trade_id'] == tid), None)
                    if closed:
                        self._write_csv({**closed, 'action': 'CLOSE',
                                         'ts': datetime.now(timezone.utc).isoformat()}, 'orders')
                elif res['action'] == 'close_tp1':
                    # Partial close: TP1 pnl on tp1_lots
                    tp1_pnl  = res.get('pnl_pips', 0.0)
                    tp1_lots = res.get('close_lots', lots_before * 0.7)
                    self._update_virtual_balance(tp1_pnl, tp1_lots)

        # ── Open new trade ────────────────────────────────────────────
        new_trade = None
        if sig.gate_passed and _trade_allowed:
            if self._traded_this_bar:
                logger.info("SKIP: already traded this bar")
                self.sig_counts['HOLD'] += 1
            elif self._direction_locked(sig.direction):
                logger.info(f"SKIP: direction lock — {sig.direction} already open")
                self.sig_counts['HOLD'] += 1
            else:
                self.sig_counts[sig.direction] += 1
                new_trade = self.mgr.open_trade(
                    sig, price, self._display_balance(), self.symbol, is_news)
                if new_trade:
                    self._traded_this_bar = True
                    self.engine.notify_trade_opened()
                    self._log_open(new_trade, sig)
        elif sig.gate_passed and not _trade_allowed:
            self.sig_counts['HOLD'] += 1
        else:
            self.sig_counts['HOLD'] += 1
            if _trade_allowed:
                self.gate_counts[sig.gate_rejected] += 1

        perf = self.mgr.summary()
        disp_balance = self._display_balance()

        # Compute live unrealised PnL for all open positions
        positions = _open_positions_list(self.mgr, price)

        row = {
            'ts':               str(ts),
            'bar':              self.bar_count,
            'bar_date':         bar_date,
            'price':            round(price, 5),
            'open':             round(float(df['open'].iloc[-1]), 5),
            'high':             round(float(df['high'].iloc[-1]), 5),
            'low':              round(float(df['low'].iloc[-1]), 5),
            'close':            round(float(df['close'].iloc[-1]), 5),
            'atr':              round(atr, 1),
            'spread':           round(spread_pips, 2),
            'is_news':          int(is_news),
            'regime':           REGIME.get(regime, '?'),
            'regime_id':        regime,
            'regime_threshold': sig.regime_threshold,
            'gmm_p0':           round(float(gp[0]), 4),
            'gmm_p1':           round(float(gp[1]), 4),
            'gmm_p2':           round(float(gp[2]), 4),
            'gmm_p3':           round(float(gp[3]), 4),
            'hmm_p0':           round(float(hp[0]), 4),
            'hmm_p1':           round(float(hp[1]), 4),
            'hmm_p2':           round(float(hp[2]), 4),
            'hmm_p3':           round(float(hp[3]), 4),
            'gmm_conf':         round(float(gp.max()), 4),
            'hmm_conf':         round(float(hp.max()), 4),
            'size_mult':        round(sig.size_mult, 3),
            'signal':           sig.direction,
            'score':            round(sig.score, 4),
            'buy_p':            round(sig.cat_buy_prob, 4),
            'sell_p':           round(sig.cat_sell_prob, 4),
            'lr_buy':           0.0,
            'edge':             round(sig.expected_edge, 1),
            'tp_pip':           round(sig.tp_pips, 1),
            'sl_pip':           round(sig.sl_pips, 1),
            'quality':          self._quality(sig.score),
            'gate':             sig.gate_rejected,
            'open_trades':      len(self.mgr.open_trades),
            'total_pnl':        round(perf.get('total_pnl', 0), 1),
            'win_rate':         round(perf.get('win_rate', 0), 1),
            'daily_loss':       round(perf.get('daily_loss_pct', 0), 3),
            'trades':           perf.get('trades', 0),
            'avg_ppt':          round(perf.get('avg_ppt', 0), 1),
            'pf':               round(perf.get('pf', 0), 2),
        }

        if _write_signals:
            self._write_csv(row, 'signals')

        _atomic_json({
            **row,
            'gmm_probs':       gp.tolist(),
            'hmm_probs':       hp.tolist(),
            'balance':         round(disp_balance, 2),
            'mt5_balance':     round(self.balance, 2),
            'virtual_balance': round(self.virtual_balance, 2),
            'open_positions':  positions,
            'closed_trades':   self.mgr.closed_trades[-50:],
            'last_ml_ts':      self._last_ml_ts,
            'inference_count': self.inference_count,
            'signal_counts':   dict(self.sig_counts),
            'gate_counts':     dict(self.gate_counts),
            'traded_this_bar': self._traded_this_bar,
        })

        if self.bar_count % 50 == 0 and _trade_allowed:
            logger.info(
                f"=== BAR {self.bar_count} | "
                f"sig={dict(self.sig_counts)} | "
                f"gates={dict(self.gate_counts)} | {perf} | "
                f"virtual_bal=${self.virtual_balance:,.2f} ===")

        return {
            'status': 'OK', 'signal': sig,
            'new_trade': new_trade, 'trade_actions': actions,
            'log': row, 'regime': regime,
            'gmm_probs': gp, 'hmm_probs': hp,
        }

    def _quality(self, score: float) -> str:
        if score >= 0.80: return 'STRONG'
        if score >= 0.65: return 'NORMAL'
        return 'WEAK'

    def _log_open(self, t, sig):
        o = {
            'action':    'OPEN',
            'trade_id':  t.trade_id,
            'direction': 'BUY' if t.direction == 1 else 'SELL',
            'symbol':    t.symbol,
            'lots':      t.lot_size,
            'entry':     round(t.entry_price, 5),
            'sl':        round(t.sl_price, 5),
            'sl_pips':   round(t.sl_pips, 1),
            'tp1':       round(t.tp1_price, 5),
            'tp_pips':   round(t.tp_pips, 1),
            'slippage':  round(t.slippage_pips, 2),
            'score':     round(sig.score, 4),
            'regime':    REGIME.get(sig.regime, '?'),
            'edge':      round(sig.expected_edge, 1),
            'quality':   self._quality(sig.score),
            'size_mult': sig.size_mult,
            'pnl_pips':  0.0,
            'reason':    'OPEN',
            'ts':        datetime.now(timezone.utc).isoformat(),
        }
        tag = '[PAPER]' if self.paper else '[LIVE]'
        logger.info(f"{tag} OPEN: direction={o['direction']} "
                    f"entry={o['entry']} sl={o['sl']} tp={o['tp1']} "
                    f"lots={o['lots']} score={o['score']}")
        self._write_csv(o, 'orders')

    def _log_close(self, trade, price: float, pnl: float, reason: str):
        d = 'BUY' if trade.direction == 1 else 'SELL'
        o = {
            'action':    'CLOSE',
            'trade_id':  trade.trade_id,
            'direction': d,
            'symbol':    trade.symbol,
            'lots':      trade.lot_size,
            'entry':     round(trade.entry_price, 5),
            'sl':        round(trade.sl_price, 5),
            'sl_pips':   round(trade.sl_pips, 1),
            'tp1':       round(trade.tp1_price, 5),
            'tp_pips':   round(trade.tp_pips, 1),
            'slippage':  round(trade.slippage_pips, 2),
            'score':     round(trade.signal_prob, 4),
            'regime':    REGIME.get(trade.regime, '?'),
            'edge':      round(trade.expected_edge, 1),
            'quality':   self._quality(trade.signal_prob),
            'size_mult': 0.0,
            'pnl_pips':  round(pnl, 2),
            'reason':    reason,
            'ts':        datetime.now(timezone.utc).isoformat(),
        }
        tag = '[PAPER]' if self.paper else '[LIVE]'
        logger.info(f"{tag} CLOSE: pnl={pnl:.1f}pip reason={reason}")
        self._write_csv(o, 'orders')

    def _log_action(self, res):
        tag = '[PAPER]' if self.paper else '[LIVE]'
        logger.info(f"{tag} {res['action']}: "
                    f"pnl={res.get('pnl_pips',0):.1f}pip "
                    f"reason={res.get('reason','')}")

    def _write_csv(self, row: dict, name: str):
        p = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m')}.csv"
        try:
            hdr = not p.exists()
            with open(p, 'a', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=row.keys())
                if hdr:
                    w.writeheader()
                w.writerow(row)
        except PermissionError:
            logger.debug(f"CSV locked: {p.name}")
        except Exception as e:
            logger.debug(f"CSV skip ({type(e).__name__}): {p.name}")
