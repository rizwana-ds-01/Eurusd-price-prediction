"""
demo_runner.py — v18.13
━━━━━━━━━━━━━━━━━━━━━━━
EDIT ONLY THESE 2 LINES:
"""
ICM_LOGIN    = 52869629
ICM_PASSWORD = "N&8b1KlvPSepC8"
# ────────────────────────────────────
ICM_SYMBOL      = "EURUSD"
ICM_DEMO_SERVER = "ICMarketsSC-Demo"
ICM_LIVE_SERVER = "ICMarketsSC-Live01"
"""
v18.13 changes:
  ✓ DIR_THR=0.55 in signal_engine — balanced BUY/SELL
  ✓ RANGE mean-reversion via price vs EMA20
  ✓ 1s price tick, 10s ML, 0.5s loop
  ✓ 1 trade per bar, direction lock
  ✓ CLOSE rows written to orders CSV
  ✓ open_positions + closed_trades in JSON
"""
import os, sys, argparse, logging, time
import numpy as np, pandas as pd

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
os.chdir(_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(_DIR, 'v1813.log'), encoding='utf-8')
    ])
logger = logging.getLogger('DemoRunner')
LOOKBACK = 200


def run_test():
    from live_pipeline import LivePipeline
    print("=" * 65)
    print("v18.13 SMOKE TEST — balanced BUY/SELL")
    print("DIR_THR=0.55 | RANGE mean-reversion | 1trade/bar")
    print("=" * 65)
    np.random.seed(42)
    n   = 600
    idx = pd.date_range('2023-01-02', periods=n, freq='4h')
    close = 1.0800 + np.cumsum(np.random.randn(n) * 0.0005 + 0.00003)
    df = pd.DataFrame({
        'open':   close - np.random.randn(n) * 0.0002,
        'high':   close + np.abs(np.random.randn(n) * 0.0007 + 0.0005),
        'low':    close - np.abs(np.random.randn(n) * 0.0007 + 0.0005),
        'close':  close,
        'volume': np.abs(4000 + np.random.randn(n) * 800),
        'spread': np.ones(n) * 1.5,
    }, index=idx)
    try:
        pipe = LivePipeline(paper_trading=True, log_subdir='test_logs')
    except FileNotFoundError as e:
        print(str(e)); return
    n_buy = 0; n_sell = 0; n_hold = 0; n_err = 0
    for i in range(LOOKBACK, n):
        res = pipe.on_bar_close(df.iloc[i-LOOKBACK:i+1], float(df['close'].iloc[i]), 1.5)
        if res.get('status') == 'ERROR': n_err += 1; continue
        sig = res['signal']
        if sig.direction == 'BUY':   n_buy  += 1
        elif sig.direction == 'SELL': n_sell += 1
        else:                         n_hold += 1
    bars = n - LOOKBACK; days = bars / 6.0
    total = n_buy + n_sell
    print(f"\nBars: {bars} (~{days:.0f} days) | Errors: {n_err}")
    print(f"BUY:  {n_buy}  ({n_buy/max(total,1)*100:.0f}%)")
    print(f"SELL: {n_sell} ({n_sell/max(total,1)*100:.0f}%)")
    print(f"HOLD: {n_hold}")
    print(f"Trades/day: {total/max(days,1):.1f}")
    print(f"Performance: {pipe.mgr.summary()}")
    if n_err == 0:
        print("\n✅ SMOKE TEST PASSED")
        if total > 0:
            ratio = n_sell / total
            if 0.2 <= ratio <= 0.8:
                print(f"✅ DIRECTION BALANCE OK — SELL ratio {ratio:.0%}")
            else:
                print(f"⚠️ Direction imbalance — SELL ratio {ratio:.0%} (expected 20-80%)")
    else:
        print(f"\n❌ {n_err} errors")


def run_validate(csv_path: str, n_bars: int = 0):
    from live_pipeline import LivePipeline
    print("=" * 65)
    print("v18.13 VALIDATION")
    print("=" * 65)
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(_DIR, csv_path)
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}"); return
    df = pd.read_csv(csv_path, parse_dates=['time'], index_col='time')
    df.columns = [c.lower() for c in df.columns]
    for c in ['open','high','low','close']:
        if c not in df.columns: print(f"❌ Missing: {c}"); return
    if 'volume' not in df.columns: df['volume'] = 5000.0
    if 'spread' not in df.columns: df['spread'] = 1.5
    if n_bars > 0: df = df.tail(n_bars + LOOKBACK)
    print(f"Data: {len(df)} bars | {df.index[0].date()} → {df.index[-1].date()}")
    try:
        pipe = LivePipeline(paper_trading=True, log_subdir='logs')
    except FileNotFoundError as e:
        print(str(e)); return
    sigs = {'BUY':0,'SELL':0,'HOLD':0}
    for i in range(LOOKBACK, len(df)):
        is_news = bool(df['is_news_event'].iloc[i]) if 'is_news_event' in df.columns else False
        res = pipe.on_bar_close(df.iloc[i-LOOKBACK:i+1], float(df['close'].iloc[i]),
                                float(df['spread'].iloc[i]), is_news)
        if res.get('status') == 'ERROR': continue
        sigs[res['signal'].direction] += 1
        if (i - LOOKBACK + 1) % 500 == 0:
            print(f"  {i-LOOKBACK+1}/{len(df)-LOOKBACK} | sigs={sigs} | {pipe.mgr.summary()}")
    n_total = len(df) - LOOKBACK; days = n_total / 6.0
    total = sigs['BUY'] + sigs['SELL']
    print(f"\nBars: {n_total} | Days: {days:.0f}")
    print(f"BUY: {sigs['BUY']} | SELL: {sigs['SELL']} | HOLD: {sigs['HOLD']}")
    print(f"Trades/day: {total/max(days,1):.2f}")
    perf = pipe.mgr.summary()
    for k, v in perf.items(): print(f"  {k}: {v}")


def run_demo():
    from live_pipeline import LivePipeline
    from icm_broker import ICMBroker
    print("=" * 65)
    print(f"v18.13 DEMO | {ICM_SYMBOL} H4 | PAPER TRADING")
    print("Balanced BUY/SELL | RANGE mean-reversion | 1trade/bar")
    print("Price: 1s | ML: 10s | Trade: H4 close only")
    print("=" * 65)
    try:
        pipe = LivePipeline(paper_trading=True, log_subdir='demo_logs')
    except FileNotFoundError as e:
        print(str(e)); return
    broker = ICMBroker(login=ICM_LOGIN, password=ICM_PASSWORD,
                       demo=True, symbol=ICM_SYMBOL)
    print("Connecting to MT5...")
    if not broker.connect():
        print("  code=-6 or 10004 → MT5 not open")
        print("  code=10013       → wrong login/password")
        return
    balance = broker.get_balance()
    pipe.set_balance(balance)
    last_bar = None; last_ml_t = 0.0; last_price_t = 0.0; consecutive_errors = 0
    print(f"\n🟢 Running | {ICM_SYMBOL} H4 | Ctrl+C to stop")
    print(f"   Dashboard: http://localhost:8501\n")
    while True:
        try:
            now = time.time()
            if not broker.is_connected():
                logger.warning("Disconnected — reconnecting...")
                if not broker.reconnect(max_attempts=10, wait_seconds=60):
                    time.sleep(300); continue
                balance = broker.get_balance(); pipe.set_balance(balance)
            if not broker.is_market_open(ICM_SYMBOL):
                logger.info("Market closed — waiting 10min")
                time.sleep(600); continue
            if now - last_price_t >= 1.0:
                try:
                    tick_df = broker.get_bars(timeframe='H4', count=3)
                    if not tick_df.empty:
                        pipe.update_price_only(float(tick_df['close'].iloc[-1]),
                                               broker.get_spread(ICM_SYMBOL))
                except Exception as e:
                    logger.debug(f"Price tick: {e}")
                last_price_t = now
            if now - last_ml_t >= 10.0:
                df = broker.get_bars(timeframe='H4', count=250)
                if df.empty: time.sleep(0.5); continue
                price = float(df['close'].iloc[-1])
                spread = broker.get_spread(ICM_SYMBOL)
                cur_bar = df.index[-2]
                is_new_bar = (cur_bar != last_bar)
                consecutive_errors = 0
                if is_new_bar:
                    balance = broker.get_balance(); pipe.set_balance(balance)
                res = pipe.on_bar_update(df.iloc[:-1], price, spread, is_new_bar)
                sig = res.get('signal')
                if is_new_bar:
                    last_bar = cur_bar
                    bar_n = res.get('log', {}).get('bar', '?')
                    if sig and sig.gate_passed:
                        print(f"\n🔔 BAR {bar_n} | {sig.direction} | "
                              f"score={sig.score:.4f} | "
                              f"p={max(sig.cat_buy_prob,sig.cat_sell_prob):.4f} | "
                              f"TP={sig.tp_pips:.0f} SL={sig.sl_pips:.0f}pip | "
                              f"{sig.regime_name}")
                    else:
                        if sig:
                            print(f"\n   BAR {bar_n} | HOLD | {sig.gate_rejected} | "
                                  f"score={sig.score:.4f} | {sig.regime_name}")
                    for a in res.get('trade_actions', []):
                        icon = '⚡' if 'SIGNAL' in a.get('reason','') else '📌'
                        print(f"   {icon} {a.get('reason','')} | pnl={a.get('pnl_pips',0):.1f}pip")
                    perf = pipe.mgr.summary()
                    print(f"   📊 trades={perf.get('trades',0)} "
                          f"open={len(pipe.mgr.open_trades)} "
                          f"pnl={perf.get('total_pnl',0):.1f}pip "
                          f"wr={perf.get('win_rate',0):.0f}%")
                else:
                    if sig:
                        print(f"\r   📡 {sig.direction} sc={sig.score:.4f} "
                              f"p={max(sig.cat_buy_prob,sig.cat_sell_prob):.4f} "
                              f"{sig.regime_name} [{price:.5f}]    ",
                              end='', flush=True)
                last_ml_t = now
            time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n\nStopped.")
            print(f"Performance: {pipe.mgr.summary()}")
            broker.disconnect(); break
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"Error #{consecutive_errors}: {e}", exc_info=True)
            time.sleep(300 if consecutive_errors >= 5 else 60)
            if consecutive_errors >= 5: consecutive_errors = 0


def run_live():
    from live_pipeline import LivePipeline
    from icm_broker import ICMBroker
    print("=" * 65)
    print(f"⚠️  LIVE TRADING — REAL MONEY | {ICM_LIVE_SERVER}")
    print("=" * 65)
    if input("Type  GO LIVE  to confirm: ").strip() != 'GO LIVE':
        print("Cancelled."); return
    broker = ICMBroker(login=ICM_LOGIN, password=ICM_PASSWORD,
                       demo=False, symbol=ICM_SYMBOL)
    if not broker.connect(): print("❌ Cannot connect"); return
    balance = broker.get_balance()
    print(f"\nLive balance: ${balance:,.2f}")
    try:
        pipe = LivePipeline(paper_trading=False, log_subdir='live_logs')
    except FileNotFoundError as e:
        print(str(e)); broker.disconnect(); return
    pipe.set_balance(balance)
    last_bar = None; ticket_map = {}; last_ml_t = 0.0; last_price_t = 0.0
    print(f"\n🔴 LIVE | {ICM_SYMBOL} H4 | Ctrl+C to stop")
    while True:
        try:
            now = time.time()
            if not broker.is_connected():
                if not broker.reconnect(10, 60): time.sleep(300); continue
                pipe.set_balance(broker.get_balance())
            if not broker.is_market_open(ICM_SYMBOL): time.sleep(600); continue
            if now - last_price_t >= 1.0:
                try:
                    tick_df = broker.get_bars(timeframe='H4', count=3)
                    if not tick_df.empty:
                        pipe.update_price_only(float(tick_df['close'].iloc[-1]),
                                               broker.get_spread(ICM_SYMBOL))
                except Exception: pass
                last_price_t = now
            if now - last_ml_t >= 10.0:
                df = broker.get_bars(timeframe='H4', count=250)
                if df.empty: time.sleep(0.5); continue
                price = float(df['close'].iloc[-1])
                spread = broker.get_spread(ICM_SYMBOL)
                cur_bar = df.index[-2]; is_new_bar = (cur_bar != last_bar)
                if is_new_bar: pipe.set_balance(broker.get_balance())
                res = pipe.on_bar_update(df.iloc[:-1], price, spread, is_new_bar)
                sig = res.get('signal')
                if is_new_bar:
                    last_bar = cur_bar
                    if sig and sig.gate_passed:
                        t = res.get('new_trade')
                        if t:
                            d = 'BUY' if t.direction == 1 else 'SELL'
                            ticket = broker.open_order(d, t.lot_size, t.sl_price,
                                                       t.tp1_price, comment='v1813')
                            if ticket: ticket_map[t.trade_id] = ticket
                    for action in res.get('trade_actions', []):
                        for tid, ticket in list(ticket_map.items()):
                            t = pipe.mgr.open_trades.get(tid)
                            closed = next((x for x in pipe.mgr.closed_trades
                                           if x['trade_id'] == tid), None)
                            if closed:
                                broker.close_position(ticket, closed['lots'], closed['direction'])
                                del ticket_map[tid]
                            elif action.get('action') == 'close_tp1' and t:
                                d = 'BUY' if t.direction == 1 else 'SELL'
                                broker.partial_close(ticket, t.tp1_lots, d)
                                broker.modify_sl(ticket, t.entry_price)
                last_ml_t = now
            time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopped."); print(pipe.mgr.summary()); broker.disconnect(); break
        except Exception as e:
            logger.error(f"Live error: {e}", exc_info=True); time.sleep(60)


if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='v18.13 EUR/USD H4 System')
    pa.add_argument('--mode', choices=['test','validate','demo','live'], default='test')
    pa.add_argument('--csv', default='')
    pa.add_argument('--bars', type=int, default=0)
    a = pa.parse_args()
    if   a.mode == 'test':     run_test()
    elif a.mode == 'validate':
        if not a.csv: print("❌ --csv required"); sys.exit(1)
        run_validate(a.csv, a.bars)
    elif a.mode == 'demo':     run_demo()
    elif a.mode == 'live':     run_live()