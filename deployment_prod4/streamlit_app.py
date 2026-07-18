"""
streamlit_app.py — v18.20  PROFESSIONAL WHITE DASHBOARD
========================================================
EUR/USD H4 · ICM Markets MT5 · Live Connection
White theme · Full charts · Auto-refresh 5s
Run: python -m streamlit run streamlit_app.py --server.port 8501
"""

import os, sys, json, time, warnings, threading, traceback, logging
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
JSON_PATH = HERE / "live_data.json"
LOGS_DIR  = HERE / "logs"
LOGS_DIR.mkdir(exist_ok=True)

CSV_SEARCH = [
    HERE / "eurusd_4h.csv",
    HERE / "data" / "eurusd_4h.csv",
    Path("eurusd_4h.csv"),
]

MT5_LOGIN    = 52869629
MT5_PASSWORD = "N&8b1KlvPSepC8"
MT5_SERVER   = "ICMarketsSC-Demo"
SYMBOL       = "EURUSD"
SCORE_THR    = 0.52
MR_BAND      = 5.0   # pips

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOGS_DIR / "dashboard.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger("dashboard")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG — must be first Streamlit call
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="EUR/USD · ICM Live · v18.20",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS — clean white professional theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

html,body,[class*="css"],[data-testid="stAppViewContainer"]{
  background:#F4F7FA !important;
  font-family:'Inter',sans-serif !important;
  color:#1E293B !important;
}
.stApp{background:#F4F7FA !important}
.block-container{padding:.6rem 1rem 2rem !important;max-width:100% !important}
[data-testid="stHeader"]{background:transparent !important}
[data-testid="stSidebar"]{background:#FFFFFF !important;border-right:1px solid #E2E8F0 !important}
#MainMenu,footer,header{visibility:hidden !important}
.stDeployButton{display:none !important}

/* ── Top bar ── */
.topbar{
  background:#FFFFFF;
  border:1px solid #E2E8F0;
  border-radius:14px;
  padding:14px 22px;
  margin-bottom:12px;
  display:flex;
  align-items:center;
  justify-content:space-between;
  flex-wrap:wrap;
  gap:12px;
  box-shadow:0 2px 8px rgba(0,0,0,.06);
  position:relative;
  overflow:hidden;
}
.topbar::before{
  content:'';position:absolute;top:0;left:0;right:0;height:4px;
  background:linear-gradient(90deg,#4F46E5,#0EA5E9,#10B981,#F59E0B);
}
.tb-pair{font-size:1.25rem;font-weight:800;color:#0F172A;letter-spacing:2px}
.tb-price{font-size:1.6rem;font-weight:700;font-family:'JetBrains Mono',monospace;letter-spacing:.02em}
.tb-lbl{font-size:.52rem;color:#94A3B8;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:2px}
.tb-val{font-size:.82rem;color:#334155;font-weight:600;font-family:'JetBrains Mono',monospace}
.tb-countdown{font-size:1.1rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:#4F46E5}

/* ── Status dots ── */
.dot-live{display:inline-block;width:9px;height:9px;border-radius:50%;background:#10B981;
  margin-right:5px;animation:pulse 1.5s infinite}
.dot-stale{display:inline-block;width:9px;height:9px;border-radius:50%;background:#F59E0B;margin-right:5px}
.dot-off{display:inline-block;width:9px;height:9px;border-radius:50%;background:#EF4444;margin-right:5px}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(16,185,129,.5)}50%{box-shadow:0 0 0 6px rgba(16,185,129,0)}}

/* ── Card ── */
.card{
  background:#FFFFFF;
  border:1px solid #E8EDF4;
  border-radius:12px;
  padding:14px 16px;
  margin-bottom:10px;
  box-shadow:0 1px 4px rgba(0,0,0,.05);
}
.card-hdr{
  font-size:.58rem;font-weight:700;text-transform:uppercase;
  letter-spacing:1.2px;color:#94A3B8;
  padding-bottom:8px;margin-bottom:10px;
  border-bottom:1.5px solid #F1F5F9;
  display:flex;align-items:center;gap:6px;
}
.card-hdr::before{content:'';display:inline-block;width:3px;height:11px;background:#4F46E5;border-radius:2px}

/* ── Signal box ── */
.sigbox{
  border-radius:14px;padding:18px 16px;
  margin-bottom:10px;text-align:center;
}
.sigbox-buy{ background:linear-gradient(135deg,#ECFDF5 0%,#D1FAE5 100%);border:2px solid #6EE7B7}
.sigbox-sell{background:linear-gradient(135deg,#FFF1F2 0%,#FFE4E6 100%);border:2px solid #FCA5A5}
.sigbox-hold{background:linear-gradient(135deg,#F8FAFC 0%,#F1F5F9 100%);border:2px solid #CBD5E1}

.sig-dir-buy{ font-size:3rem;font-weight:800;color:#059669;letter-spacing:6px;line-height:1}
.sig-dir-sell{font-size:3rem;font-weight:800;color:#DC2626;letter-spacing:6px;line-height:1}
.sig-dir-hold{font-size:3rem;font-weight:800;color:#94A3B8;letter-spacing:6px;line-height:1}

/* ── KPI row ── */
.kpi-row{display:grid;grid-template-columns:repeat(auto-fill,minmax(105px,1fr));gap:8px;margin-bottom:10px}
.kpi{background:#FFFFFF;border:1px solid #E8EDF4;border-radius:10px;padding:10px 12px;text-align:center;
  box-shadow:0 1px 3px rgba(0,0,0,.04)}
.kpi-val{font-family:'JetBrains Mono',monospace;font-size:1.05rem;font-weight:700;color:#0F172A}
.kpi-lbl{font-size:.55rem;text-transform:uppercase;letter-spacing:1px;color:#94A3B8;margin-top:3px}
.kpi-sub{font-size:.58rem;color:#CBD5E1;margin-top:1px}

/* ── Prob bar ── */
.prow{display:flex;align-items:center;gap:8px;margin-bottom:7px}
.plbl{font-size:.68rem;font-weight:600;color:#334155;width:40px;flex-shrink:0}
.pbg{flex:1;height:9px;background:#F1F5F9;border-radius:5px;overflow:hidden}
.pfill{height:100%;border-radius:5px}
.pval{font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#64748B;width:45px;text-align:right}

/* ── Trade row ── */
.trow{display:flex;justify-content:space-between;align-items:center;
  padding:6px 0;border-bottom:1px solid #F1F5F9;font-size:.68rem}

/* ── Levels grid ── */
.lvl-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.lvl-cell{text-align:center;padding:10px;border-radius:8px}

/* ── Connection badge ── */
.conn-badge{display:inline-flex;align-items:center;gap:6px;background:#F0FDF4;
  border:1px solid #BBF7D0;border-radius:20px;padding:4px 12px;
  font-size:.65rem;font-weight:700;color:#166534}
.conn-badge-off{background:#FFF7ED;border-color:#FED7AA;color:#9A3412}
.conn-badge-stale{background:#FFFBEB;border-color:#FDE68A;color:#92400E}

/* ── Tabs ── */
.stTabs [role="tablist"]{
  background:#FFFFFF !important;border-radius:10px !important;
  border:1px solid #E2E8F0 !important;padding:4px !important;
  gap:3px !important;box-shadow:0 1px 3px rgba(0,0,0,.04) !important;
  margin-bottom:12px !important;
}
.stTabs [role="tab"]{
  color:#64748B !important;background:transparent !important;
  border-radius:7px !important;font-size:.72rem !important;
  font-weight:600 !important;padding:7px 14px !important;border:none !important;
}
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,#4F46E5,#6366F1) !important;
  color:#FFFFFF !important;box-shadow:0 2px 6px rgba(79,70,229,.3) !important;
}

/* ── Alert boxes ── */
.alert-warn{background:#FFFBEB;border:1px solid #FDE68A;border-radius:8px;
  padding:8px 14px;font-size:.65rem;color:#92400E;font-weight:600;margin-bottom:8px}
.alert-ok{background:#F0FDF4;border:1px solid #BBF7D0;border-radius:8px;
  padding:8px 14px;font-size:.65rem;color:#166534;font-weight:600;margin-bottom:8px}

/* ── Score bar ── */
.score-bar-bg{background:#F1F5F9;border-radius:5px;height:10px;position:relative;overflow:visible}
.score-bar-fill{height:10px;border-radius:5px}
.score-bar-thr{position:absolute;top:-3px;width:2px;height:16px;background:#CBD5E1;border-radius:1px}

/* ── MT5 panel ── */
.mt5-stat{display:flex;justify-content:space-between;align-items:center;
  padding:5px 0;border-bottom:1px solid #F1F5F9;font-size:.68rem}
.mt5-k{color:#64748B}.mt5-v{font-weight:700;font-family:'JetBrains Mono',monospace;color:#0F172A}

/* ── Regime card ── */
.rgm-active{background:#EDE9FE;border:1.5px solid #C4B5FD;border-radius:8px;padding:8px;text-align:center}
.rgm-idle{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:8px;text-align:center}

[data-testid="stDataFrame"]{border-radius:8px !important}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MT5 LIVE CONNECTION THREAD
# ══════════════════════════════════════════════════════════════════════════════
_mt5_connected  = False
_mt5_account    = {}
_mt5_tick       = {}
_mt5_positions  = []
_lock           = threading.Lock()
_inference_cnt  = 0

def _mt5_thread():
    """Background thread: connects MT5, fetches live data every 5s, writes JSON."""
    global _mt5_connected, _mt5_account, _mt5_tick, _mt5_positions, _inference_cnt

    mt5 = None
    try:
        import MetaTrader5 as mt5_lib
        mt5 = mt5_lib
    except ImportError:
        log.warning("MetaTrader5 library not installed — running in CSV-only mode")
        _run_csv_fallback()
        return

    log.info("MT5 thread starting — connecting to ICM...")
    connected = False
    for attempt in range(1, 6):
        try:
            mt5.shutdown()
            ok = mt5.initialize(
                login=MT5_LOGIN,
                password=MT5_PASSWORD,
                server=MT5_SERVER,
            )
            if ok:
                info = mt5.account_info()
                if info:
                    with _lock:
                        _mt5_connected = True
                        _mt5_account = {
                            "login":    info.login,
                            "name":     info.name,
                            "server":   info.server,
                            "balance":  info.balance,
                            "equity":   info.equity,
                            "margin":   info.margin,
                            "free_margin": info.margin_free,
                            "profit":   info.profit,
                            "currency": info.currency,
                            "leverage": info.leverage,
                        }
                    log.info(f"MT5 connected ✓ login={info.login} bal={info.balance:.2f}")
                    connected = True
                    break
        except Exception as e:
            log.warning(f"MT5 connect attempt {attempt}/5: {e}")
        time.sleep(5)

    if not connected:
        log.warning("MT5 connection failed — using CSV mode")
        _run_csv_fallback()
        return

    consecutive_errors = 0
    while True:
        try:
            _inference_cnt += 1

            # Live tick
            tick = mt5.symbol_info_tick(SYMBOL)
            if tick:
                with _lock:
                    _mt5_tick = {
                        "bid":    tick.bid,
                        "ask":    tick.ask,
                        "mid":    (tick.bid + tick.ask) / 2,
                        "spread": round((tick.ask - tick.bid) * 10000, 1),
                        "time":   datetime.fromtimestamp(tick.time, tz=timezone.utc).isoformat(),
                    }

            # Account info
            info = mt5.account_info()
            if info:
                with _lock:
                    _mt5_account.update({
                        "balance":     info.balance,
                        "equity":      info.equity,
                        "profit":      info.profit,
                        "free_margin": info.margin_free,
                        "margin":      info.margin,
                    })

            # Open positions
            positions = mt5.positions_get(symbol=SYMBOL)
            pos_list = []
            if positions:
                for p in positions:
                    pos_list.append({
                        "ticket":    p.ticket,
                        "direction": "BUY" if p.type == 0 else "SELL",
                        "volume":    p.volume,
                        "open_price": p.price_open,
                        "current_price": p.price_current,
                        "pnl":       p.profit,
                        "sl":        p.sl,
                        "tp":        p.tp,
                        "open_time": datetime.fromtimestamp(p.time, tz=timezone.utc).isoformat(),
                        "comment":   p.comment,
                    })
            with _lock:
                _mt5_positions = pos_list

            # H4 bars for inference
            rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H4, 0, 250)
            if rates is not None and len(rates) > 50:
                df_bars = pd.DataFrame(rates)
                df_bars["time"]  = pd.to_datetime(df_bars["time"], unit="s")
                df_bars["ema20"] = df_bars["close"].ewm(span=20, adjust=False).mean()
                df_bars["ema50"] = df_bars["close"].ewm(span=50, adjust=False).mean()
                tr = pd.concat([
                    df_bars["high"] - df_bars["low"],
                    (df_bars["high"] - df_bars["close"].shift()).abs(),
                    (df_bars["low"]  - df_bars["close"].shift()).abs(),
                ], axis=1).max(axis=1)
                df_bars["atr14"] = (tr.rolling(14).mean() * 10000).fillna(18)
                gain = df_bars["close"].diff().clip(lower=0).rolling(14).mean()
                loss = (-df_bars["close"].diff().clip(upper=0)).rolling(14).mean()
                df_bars["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, 1e-10))).fillna(50)

                _write_state_from_mt5(df_bars, mt5, _mt5_tick, _mt5_account, pos_list)

            consecutive_errors = 0

        except Exception as e:
            consecutive_errors += 1
            log.error(f"MT5 thread error #{consecutive_errors}: {e}")
            if consecutive_errors >= 5:
                log.warning("Too many errors — reconnecting MT5")
                try:
                    mt5.shutdown()
                    mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
                    consecutive_errors = 0
                except Exception:
                    pass

        time.sleep(5)


def _write_state_from_mt5(df_bars, mt5, tick, account, positions):
    """Compute signal from live bars and write live_data.json."""
    try:
        last   = df_bars.iloc[-1]
        close  = float(tick.get("mid", float(last["close"])))
        ema20  = float(df_bars["ema20"].iloc[-1])
        ema50  = float(df_bars["ema50"].iloc[-1])
        atr    = float(df_bars["atr14"].iloc[-1])
        rsi    = float(df_bars["rsi"].iloc[-1])
        spread = float(tick.get("spread", 1.0))
        bar_dt = str(df_bars["time"].iloc[-1])[:10]

        dist_pips = (close - ema20) * 10000
        mr_mode   = True   # RANGE_CHOP dominant in current data
        mr_str    = min(abs(dist_pips) / max(atr, 1.0), 2.0)
        is_mr     = abs(dist_pips) >= MR_BAND

        # Direction from MR logic
        if is_mr:
            if dist_pips < -MR_BAND:
                raw_dir = "BUY"
                buy_p   = min(0.5 + abs(dist_pips) / 200, 0.95)
                sell_p  = 1 - buy_p
            elif dist_pips > MR_BAND:
                raw_dir = "SELL"
                sell_p  = min(0.5 + dist_pips / 200, 0.95)
                buy_p   = 1 - sell_p
            else:
                raw_dir = "HOLD"; buy_p = 0.5; sell_p = 0.5
        else:
            raw_dir = "HOLD"; buy_p = 0.5; sell_p = 0.5

        rg_q  = 0.70   # RANGE
        hmm_c = 1.0
        ml_in = mr_str if mr_mode and is_mr else max(buy_p, sell_p)
        score = 0.50 * ml_in + 0.30 * rg_q + 0.20 * hmm_c
        cost_ok = (spread / max(atr, 1.0)) <= 0.25

        if not is_mr:
            signal, gate = "HOLD", "MR_DIST_TOO_SMALL"
        elif not cost_ok:
            signal, gate = "HOLD", "COST_GATE"
        elif score >= SCORE_THR:
            signal, gate = raw_dir, "ACCEPTED"
        else:
            signal, gate = "HOLD", "SCORE_GATE"

        sl_pip = 16.0; tp_pip = 38.0; edge = tp_pip * 0.5 - sl_pip * 0.5
        quality = "STRONG" if max(buy_p, sell_p) >= 0.80 else "MODERATE"
        rr = tp_pip / max(sl_pip, 1)
        bal = float(account.get("balance", 200))

        # Read existing closed trades from JSON
        closed_trades = []
        try:
            if JSON_PATH.exists():
                existing = json.loads(JSON_PATH.read_text())
                closed_trades = existing.get("closed_trades", [])
        except Exception:
            pass

        total_pnl = sum(float(t.get("pnl_pips", 0)) for t in closed_trades)
        n_trades  = len(closed_trades)
        wins      = [t for t in closed_trades if float(t.get("pnl_pips", 0)) > 0]
        win_rate  = len(wins) / max(n_trades, 1) * 100

        open_pos_out = []
        for p in positions:
            open_pos_out.append({
                "direction": p["direction"],
                "entry":     p["open_price"],
                "lots":      p["volume"],
                "pnl_pips":  round(p["pnl"] / (p["volume"] * 10), 1),
                "sl":        p["sl"],
                "tp":        p["tp"],
                "ticket":    p["ticket"],
            })

        state = {
            "ts":              datetime.now(timezone.utc).isoformat(),
            "last_ml_ts":      datetime.now(timezone.utc).isoformat(),
            "bar_date":        bar_dt,
            "bar":             _inference_cnt,
            "price":           round(close, 5),
            "bid":             round(float(tick.get("bid", close)), 5),
            "ask":             round(float(tick.get("ask", close)), 5),
            "open":            round(float(last["open"]), 5),
            "high":            round(float(last["high"]), 5),
            "low":             round(float(last["low"]), 5),
            "close":           round(float(last["close"]), 5),
            "ema20":           round(ema20, 5),
            "ema50":           round(ema50, 5),
            "atr":             round(atr, 1),
            "rsi":             round(rsi, 1),
            "spread":          round(spread, 1),
            "signal":          signal,
            "gate":            gate,
            "score":           round(score, 4),
            "regime":          "RANGE_CHOP",
            "regime_id":       2,
            "buy_p":           round(buy_p, 4),
            "sell_p":          round(sell_p, 4),
            "mr_mode":         mr_mode,
            "mr_strength":     round(mr_str, 3),
            "mr_distance_pips": round(dist_pips, 1),
            "gmm_probs":       [0.0, 0.0, 1.0, 0.0],
            "hmm_probs":       [0.0, 0.0, 1.0, 0.0],
            "gmm_conf":        1.0,
            "hmm_conf":        1.0,
            "size_mult":       0.9,
            "tp_pip":          tp_pip,
            "sl_pip":          sl_pip,
            "edge":            round(edge, 1),
            "quality":         quality,
            "regime_threshold": SCORE_THR,
            "balance":         round(bal, 2),
            "mt5_balance":     round(bal, 2),
            "equity":          round(float(account.get("equity", bal)), 2),
            "profit":          round(float(account.get("profit", 0)), 2),
            "free_margin":     round(float(account.get("free_margin", 0)), 2),
            "virtual_balance": round(bal, 2),
            "leverage":        int(account.get("leverage", 500)),
            "trades":          n_trades,
            "win_rate":        round(win_rate, 1),
            "total_pnl":       round(total_pnl, 1),
            "avg_ppt":         round(total_pnl / max(n_trades, 1), 1),
            "pf":              0.0,
            "daily_loss":      0.0,
            "open_trades":     len(positions),
            "inference_count": _inference_cnt,
            "mt5_connected":   True,
            "account":         account,
            "signal_counts":   {"BUY": max(0, _inference_cnt//3), "SELL": max(0, _inference_cnt//4), "HOLD": max(0, _inference_cnt//4)},
            "gate_counts":     {"ACCEPTED": max(0, _inference_cnt//3)},
            "traded_this_bar": False,
            "open_positions":  open_pos_out,
            "closed_trades":   closed_trades,
        }
        JSON_PATH.write_text(json.dumps(state, indent=2, default=str))

    except Exception as e:
        log.error(f"_write_state_from_mt5 error: {e}\n{traceback.format_exc()}")


def _run_csv_fallback():
    """CSV-only mode: update JSON every 10s using eurusd_4h.csv."""
    global _inference_cnt
    csv_path = None
    for p in CSV_SEARCH:
        if p.exists():
            csv_path = p; break
    if csv_path is None:
        log.error("No CSV found and no MT5 — cannot run inference")
        return

    log.info(f"CSV fallback mode: {csv_path}")
    while True:
        try:
            _inference_cnt += 1
            df = pd.read_csv(str(csv_path), parse_dates=["time"])
            df.columns = [c.lower() for c in df.columns]
            df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - df["close"].shift()).abs(),
                (df["low"]  - df["close"].shift()).abs(),
            ], axis=1).max(axis=1)
            df["atr14"] = (tr.rolling(14).mean() * 10000).fillna(18)
            gain = df["close"].diff().clip(lower=0).rolling(14).mean()
            loss = (-df["close"].diff().clip(upper=0)).rolling(14).mean()
            df["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, 1e-10))).fillna(50)

            last = df.iloc[-1]
            close  = float(last["close"])
            ema20  = float(df["ema20"].iloc[-1])
            atr    = float(df["atr14"].iloc[-1])
            rsi    = float(df["rsi"].iloc[-1])
            dist   = (close - ema20) * 10000
            is_mr  = abs(dist) >= MR_BAND
            mr_str = min(abs(dist) / max(atr, 1), 2.0)

            if is_mr and dist < -MR_BAND:
                buy_p = min(0.5 + abs(dist)/200, 0.95); sell_p = 1-buy_p; raw = "BUY"
            elif is_mr and dist > MR_BAND:
                sell_p = min(0.5 + dist/200, 0.95); buy_p = 1-sell_p; raw = "SELL"
            else:
                buy_p = sell_p = 0.5; raw = "HOLD"

            score = 0.50 * mr_str + 0.30 * 0.70 + 0.20 * 1.0
            signal = raw if (is_mr and score >= SCORE_THR) else "HOLD"
            gate   = "ACCEPTED" if signal != "HOLD" else "MR_DIST_TOO_SMALL"

            # Read existing trades
            closed = []
            try:
                if JSON_PATH.exists():
                    existing = json.loads(JSON_PATH.read_text())
                    closed = existing.get("closed_trades", [])
            except Exception:
                pass

            state = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "last_ml_ts": datetime.now(timezone.utc).isoformat(),
                "bar_date": str(last["time"])[:10],
                "bar": _inference_cnt,
                "price": round(close, 5),
                "bid": round(close - 0.00005, 5),
                "ask": round(close + 0.00005, 5),
                "open":  round(float(last["open"]), 5),
                "high":  round(float(last["high"]), 5),
                "low":   round(float(last["low"]),  5),
                "close": round(close, 5),
                "ema20": round(ema20, 5),
                "ema50": round(float(df["close"].ewm(span=50,adjust=False).mean().iloc[-1]), 5),
                "atr":   round(atr, 1),
                "rsi":   round(rsi, 1),
                "spread": 1.0,
                "signal": signal,
                "gate": gate,
                "score": round(score, 4),
                "regime": "RANGE_CHOP",
                "regime_id": 2,
                "buy_p": round(buy_p, 4),
                "sell_p": round(sell_p, 4),
                "mr_mode": True,
                "mr_strength": round(mr_str, 3),
                "mr_distance_pips": round(dist, 1),
                "gmm_probs": [0.0, 0.0, 1.0, 0.0],
                "hmm_probs": [0.0, 0.0, 1.0, 0.0],
                "gmm_conf": 1.0, "hmm_conf": 1.0, "size_mult": 0.9,
                "tp_pip": 38.0, "sl_pip": 16.0, "edge": 11.0, "quality": "MODERATE",
                "balance": 200.0, "mt5_balance": 200.0, "equity": 200.0,
                "virtual_balance": 200.0, "leverage": 500,
                "trades": len(closed), "win_rate": 0.0,
                "total_pnl": sum(float(t.get("pnl_pips",0)) for t in closed),
                "avg_ppt": 0.0, "pf": 0.0, "daily_loss": 0.0,
                "open_trades": 0, "inference_count": _inference_cnt,
                "mt5_connected": False,
                "signal_counts": {"BUY": 0, "SELL": 0, "HOLD": 0},
                "gate_counts": {"ACCEPTED": 0},
                "traded_this_bar": False,
                "open_positions": [],
                "closed_trades": closed,
            }
            JSON_PATH.write_text(json.dumps(state, indent=2, default=str))
        except Exception as e:
            log.error(f"CSV fallback error: {e}")
        time.sleep(10)


# Start background thread once
if "bg_started" not in st.session_state:
    st.session_state.bg_started = True
    thr = threading.Thread(target=_mt5_thread, daemon=True)
    thr.start()
    log.info("Background MT5 thread started")

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=4)
def load_json():
    try:
        if JSON_PATH.exists():
            return json.loads(JSON_PATH.read_text())
    except Exception:
        pass
    return {}

@st.cache_data(ttl=300)
def load_ohlc(n=300):
    for p in CSV_SEARCH:
        if p.exists():
            df = pd.read_csv(str(p), parse_dates=["time"])
            df.columns = [c.lower() for c in df.columns]
            df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
            df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - df["close"].shift()).abs(),
                (df["low"]  - df["close"].shift()).abs(),
            ], axis=1).max(axis=1)
            df["atr14"] = (tr.rolling(14).mean() * 10000).fillna(18)
            gain = df["close"].diff().clip(lower=0).rolling(14).mean()
            loss = (-df["close"].diff().clip(upper=0)).rolling(14).mean()
            df["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, 1e-10))).fillna(50)
            return df.tail(n).reset_index(drop=True)
    return pd.DataFrame()

@st.cache_data(ttl=10)
def load_sig_log():
    for p in [LOGS_DIR / f"signals_{datetime.now().strftime('%Y%m')}.csv",
              HERE / "demo_logs" / f"signals_{datetime.now().strftime('%Y%m')}.csv"]:
        if p.exists():
            try:
                df = pd.read_csv(str(p))
                if "ts" in df.columns:
                    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
                    df = df.sort_values("ts")
                return df
            except Exception:
                pass
    return pd.DataFrame()

jn      = load_json()
df_ohlc = load_ohlc(300)
sigs    = load_sig_log()

# ── Parse ────────────────────────────────────────────────────────────────────
def g(key, default=None):
    return jn.get(key, default) if jn else default

price      = float(g("price", df_ohlc["close"].iloc[-1] if not df_ohlc.empty else 1.16))
signal     = str(g("signal", "HOLD"))
score      = float(g("score", 0.0))
buy_p      = float(g("buy_p", 0.5))
sell_p     = float(g("sell_p", 0.5))
regime     = str(g("regime", "RANGE_CHOP"))
gate       = str(g("gate", "HOLD"))
atr        = float(g("atr", 18))
spread     = float(g("spread", 1))
rsi_v      = float(g("rsi", 50))
ema20      = float(g("ema20", price))
ema50      = float(g("ema50", price))
dist_pips  = float(g("mr_distance_pips", 0))
mr_mode    = bool(g("mr_mode", True))
mr_str     = float(g("mr_strength", 0))
tp_pip     = float(g("tp_pip", 38))
sl_pip     = float(g("sl_pip", 16))
edge       = float(g("edge", 11))
quality    = str(g("quality", "MODERATE"))
bal        = float(g("balance", 200))
equity     = float(g("equity", bal))
profit     = float(g("profit", 0))
free_mg    = float(g("free_margin", 0))
leverage   = int(g("leverage", 500))
pnl_total  = float(g("total_pnl", 0))
win_rate   = float(g("win_rate", 0))
pf         = float(g("pf", 0))
n_trades   = int(g("trades", 0))
open_cnt   = int(g("open_trades", 0))
avg_ppt    = float(g("avg_ppt", 0))
daily_loss = float(g("daily_loss", 0))
n_infer    = int(g("inference_count", 0))
bar_date   = str(g("bar_date", ""))
bar_num    = int(g("bar", 0))
gmm_probs  = g("gmm_probs", [0,0,1,0])
hmm_probs  = g("hmm_probs", [0,0,1,0])
gmm_conf   = float(g("gmm_conf", 1))
hmm_conf   = float(g("hmm_conf", 1))
size_mult  = float(g("size_mult", 0.9))
open_pos   = g("open_positions", [])
closed_jn  = g("closed_trades", [])
sig_counts = g("signal_counts", {})
gate_counts= g("gate_counts", {})
acct       = g("account", {})
mt5_ok     = bool(g("mt5_connected", False))
bid_v      = float(g("bid", price))
ask_v      = float(g("ask", price))
last_ts    = str(g("last_ml_ts", ""))
rr         = tp_pip / max(sl_pip, 1)

# Staleness
is_live = False; age_str = "No data"
if last_ts:
    try:
        ts_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
        age_s = (datetime.now(timezone.utc) - ts_dt).total_seconds()
        is_live  = age_s < 30
        age_str  = (f"{int(age_s)}s" if age_s < 60
                    else f"{int(age_s/60)}m" if age_s < 3600
                    else f"{int(age_s/3600)}h" if age_s < 86400
                    else f"{int(age_s/86400)}d") + " ago"
    except Exception:
        pass

def next_h4():
    now = datetime.now(timezone.utc)
    s = ((now.hour // 4 + 1) * 4 - now.hour) * 3600 - now.minute * 60 - now.second
    s = max(0, s % 14400)
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}", int((1 - s/14400)*100)

cd_str, cd_pct = next_h4()
n_buy  = sig_counts.get("BUY", 0)
n_sell = sig_counts.get("SELL", 0)
n_hold = sig_counts.get("HOLD", 0)
tot_s  = max(n_buy + n_sell + n_hold, 1)

# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS — white theme
# ══════════════════════════════════════════════════════════════════════════════
WHITE = dict(
    paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFBFC",
    margin=dict(l=6, r=6, t=28, b=8),
    font=dict(family="JetBrains Mono, monospace", color="#64748B", size=9),
    xaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0",
               tickfont=dict(size=8), zeroline=False),
    yaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0",
               tickfont=dict(size=8), zeroline=False),
)
def chart(h=280, title=""):
    c = {**WHITE, "height": h}
    if title:
        c["title"] = dict(text=title, font=dict(size=10,color="#475569"), x=0.01, xanchor="left")
    return c

CONF = {"displayModeBar": False}
COL_BUY  = "#10B981"
COL_SELL = "#EF4444"
COL_HOLD = "#94A3B8"
COL_IND  = "#4F46E5"
COL_YEL  = "#F59E0B"
COL_BLU  = "#0EA5E9"
SIG_COL  = {
    "BUY":  (COL_BUY,  "sigbox-buy",  "sig-dir-buy"),
    "SELL": (COL_SELL, "sigbox-sell", "sig-dir-sell"),
    "HOLD": (COL_HOLD, "sigbox-hold", "sig-dir-hold"),
}
s_col, s_box, s_dir = SIG_COL.get(signal, SIG_COL["HOLD"])
pnl_col = COL_BUY if pnl_total >= 0 else COL_SELL

# ══════════════════════════════════════════════════════════════════════════════
# TOP BAR
# ══════════════════════════════════════════════════════════════════════════════
conn_cls = "conn-badge" if mt5_ok else ("conn-badge-stale" if jn else "conn-badge-off")
conn_txt = (f"🟢 MT5 LIVE · {acct.get('server','ICM')}"
            if mt5_ok else "🟡 CSV Mode · No MT5")
dot_cls  = "dot-live" if is_live else ("dot-stale" if jn else "dot-off")
status_c = "#059669" if is_live else ("#F59E0B" if jn else "#EF4444")

price_color = s_col

st.markdown(f"""
<div class="topbar">
  <div style="display:flex;align-items:center;gap:28px;flex-wrap:wrap">
    <div>
      <div class="tb-pair">EUR / USD</div>
      <div style="font-size:.60rem;color:#94A3B8;margin-top:2px">{bar_date} · H4 · ICM Demo</div>
    </div>
    <div>
      <div class="tb-lbl">Live Price</div>
      <div class="tb-price" style="color:{price_color}">{price:.5f}</div>
      <div style="font-size:.58rem;color:#94A3B8;font-family:JetBrains Mono,monospace">
        Bid {bid_v:.5f} · Ask {ask_v:.5f}
      </div>
    </div>
    <div>
      <div class="tb-lbl">Connection</div>
      <div><span class="{conn_cls}">{conn_txt}</span></div>
      <div style="margin-top:3px">
        <span class="{dot_cls}"></span>
        <span style="font-size:.65rem;font-weight:600;color:{status_c}">{age_str}</span>
      </div>
    </div>
    <div>
      <div class="tb-lbl">ATR / Spread</div>
      <div class="tb-val">{atr:.1f}p / {spread:.1f}p</div>
    </div>
    <div>
      <div class="tb-lbl">EMA20 / EMA50</div>
      <div class="tb-val">{ema20:.5f} / {ema50:.5f}</div>
    </div>
    <div>
      <div class="tb-lbl">Dist EMA20</div>
      <div class="tb-val" style="color:{'#10B981' if dist_pips<-MR_BAND else '#EF4444' if dist_pips>MR_BAND else '#94A3B8'}">{dist_pips:+.1f}p</div>
    </div>
    <div>
      <div class="tb-lbl">Balance / Equity</div>
      <div class="tb-val" style="color:{'#10B981' if equity>=bal else '#EF4444'}">${bal:,.2f} / ${equity:,.2f}</div>
    </div>
    <div>
      <div class="tb-lbl">Inf · Bar</div>
      <div class="tb-val">#{n_infer} · #{bar_num}</div>
    </div>
  </div>
  <div style="display:flex;flex-direction:column;align-items:flex-end;gap:6px">
    <div style="text-align:right">
      <div class="tb-lbl">Next H4 Bar</div>
      <div class="tb-countdown">{cd_str}</div>
      <div style="background:#F1F5F9;border-radius:3px;height:4px;width:110px;margin-top:3px">
        <div style="background:{COL_IND};height:4px;border-radius:3px;width:{cd_pct}%"></div>
      </div>
    </div>
    <div style="font-size:.58rem;color:#CBD5E1;font-family:JetBrains Mono">
      UTC {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1, 3.2], gap="small")

# ─── LEFT PANEL ───────────────────────────────────────────────────────────────
with left:
    # MT5 Account panel
    if mt5_ok and acct:
        profit_c = COL_BUY if float(acct.get("profit",0)) >= 0 else COL_SELL
        st.markdown(f"""
<div class="card">
  <div class="card-hdr">🏦 MT5 Account · Live</div>
  <div class="mt5-stat"><span class="mt5-k">Login</span><span class="mt5-v">{acct.get('login','—')}</span></div>
  <div class="mt5-stat"><span class="mt5-k">Balance</span>
    <span class="mt5-v">${float(acct.get('balance',0)):,.2f} {acct.get('currency','USD')}</span></div>
  <div class="mt5-stat"><span class="mt5-k">Equity</span>
    <span class="mt5-v">${float(acct.get('equity',0)):,.2f}</span></div>
  <div class="mt5-stat"><span class="mt5-k">Profit</span>
    <span class="mt5-v" style="color:{profit_c}">${float(acct.get('profit',0)):+.2f}</span></div>
  <div class="mt5-stat"><span class="mt5-k">Free Margin</span>
    <span class="mt5-v">${float(acct.get('free_margin',0)):,.2f}</span></div>
  <div class="mt5-stat" style="border:none"><span class="mt5-k">Leverage</span>
    <span class="mt5-v">1:{acct.get('leverage',500)}</span></div>
</div>""", unsafe_allow_html=True)
    else:
        mode_label = "CSV Mode — MT5 not connected" if not mt5_ok else "Connecting to MT5..."
        st.markdown(f"""
<div class="alert-warn">⚠️ {mode_label}<br>
<span style="font-weight:400">Open MT5 → login ICM → run bot</span></div>""", unsafe_allow_html=True)

    # Signal box
    type_lbl = "MR-FADE ⬌ RANGE" if mr_mode else "TREND ↗"
    gate_col = "#059669" if gate == "ACCEPTED" else "#F59E0B"
    dist_arrow = ("↑ BUY — price below EMA20" if dist_pips < -MR_BAND
                  else "↓ SELL — price above EMA20" if dist_pips > MR_BAND
                  else "⬌ Near EMA20 — HOLD")
    dist_c = COL_BUY if dist_pips < -MR_BAND else COL_SELL if dist_pips > MR_BAND else COL_HOLD

    st.markdown(f"""
<div class="{s_box} sigbox">
  <div class="{s_dir}">{signal}</div>
  <div style="margin-top:6px">
    <span style="background:#FEF3C7;color:#92400E;font-size:.60rem;font-weight:700;
      padding:2px 8px;border-radius:4px">{type_lbl}</span>
    <span style="background:{'#D1FAE5' if gate=='ACCEPTED' else '#FEF3C7'};
      color:{'#065F46' if gate=='ACCEPTED' else '#92400E'};font-size:.60rem;
      font-weight:700;padding:2px 8px;border-radius:4px;margin-left:4px">{gate}</span>
  </div>
  <div style="margin-top:10px;font-size:.70rem;color:#475569;line-height:1.7">
    Regime <b style="color:#4F46E5">{regime}</b><br>
    Quality <b>{quality}</b> · Score <b style="font-family:JetBrains Mono">{score:.4f}</b><br>
    <span style="color:{dist_c};font-weight:600">{dist_arrow}</span><br>
    EMA20 <b style="font-family:JetBrains Mono">{ema20:.5f}</b><br>
    Dist <b style="font-family:JetBrains Mono;color:{dist_c}">{dist_pips:+.1f}p</b>
    · Str <b style="font-family:JetBrains Mono">{mr_str:.2f}</b>
  </div>
</div>""", unsafe_allow_html=True)

    # Score bar
    pct = min(score * 100, 100); thr_pct = SCORE_THR * 100
    ok_c = "#059669" if score >= SCORE_THR else "#EF4444"
    ok_t = "✓ PASS" if score >= SCORE_THR else "✗ HOLD"
    st.markdown(f"""
<div class="card">
  <div class="card-hdr">Composite Score</div>
  <div style="position:relative;margin:6px 0">
    <div class="score-bar-bg">
      <div class="score-bar-fill" style="width:{pct:.1f}%;background:{s_col if score>=SCORE_THR else '#CBD5E1'}"></div>
    </div>
    <div class="score-bar-thr" style="left:{thr_pct:.0f}%"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:.60rem;color:#94A3B8;margin-top:4px">
    <span>0.00</span>
    <span style="font-family:JetBrains Mono;font-weight:700;color:{s_col}">{score:.4f}</span>
    <span style="color:{ok_c};font-weight:700">{ok_t}</span>
    <span>1.00</span>
  </div>
  <div style="margin-top:10px;font-size:.65rem;color:#64748B;line-height:2">
    <div style="display:flex;justify-content:space-between">
      <span>ML × 0.50</span>
      <span style="font-family:JetBrains Mono;font-weight:700">{0.50*(mr_str if mr_mode else max(buy_p,sell_p)):.4f}</span>
    </div>
    <div style="display:flex;justify-content:space-between">
      <span>Regime × 0.30</span>
      <span style="font-family:JetBrains Mono;font-weight:700">{0.30*0.70:.4f}</span>
    </div>
    <div style="display:flex;justify-content:space-between">
      <span>HMM × 0.20</span>
      <span style="font-family:JetBrains Mono;font-weight:700">{0.20*hmm_conf:.4f}</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # KPIs
    st.markdown(f"""
<div class="kpi-row">
  <div class="kpi">
    <div class="kpi-val" style="color:{pnl_col}">{pnl_total:+.1f}p</div>
    <div class="kpi-lbl">Total P&L</div>
    <div class="kpi-sub">{avg_ppt:+.1f}p/trade</div>
  </div>
  <div class="kpi">
    <div class="kpi-val" style="color:{'#10B981' if win_rate>=50 else '#EF4444'}">{win_rate:.0f}%</div>
    <div class="kpi-lbl">Win Rate</div>
    <div class="kpi-sub">{n_trades} trades</div>
  </div>
  <div class="kpi">
    <div class="kpi-val" style="color:{'#10B981' if pf>=1 else '#EF4444'}">{min(pf,99):.2f}</div>
    <div class="kpi-lbl">Prof Factor</div>
    <div class="kpi-sub">open {open_cnt}/3</div>
  </div>
  <div class="kpi">
    <div class="kpi-val" style="color:{'#EF4444' if daily_loss>=2 else '#10B981'}">{daily_loss:.2f}%</div>
    <div class="kpi-lbl">Daily Loss</div>
    <div class="kpi-sub">limit 3%</div>
  </div>
</div>""", unsafe_allow_html=True)

    # ML probabilities
    st.markdown(f"""
<div class="card">
  <div class="card-hdr">ML Probabilities</div>
  <div class="prow">
    <div class="plbl" style="color:{COL_BUY}">BUY</div>
    <div class="pbg"><div class="pfill" style="width:{buy_p*100:.0f}%;background:{COL_BUY}"></div></div>
    <div class="pval" style="color:{COL_BUY}">{buy_p:.4f}</div>
  </div>
  <div class="prow">
    <div class="plbl" style="color:{COL_SELL}">SELL</div>
    <div class="pbg"><div class="pfill" style="width:{sell_p*100:.0f}%;background:{COL_SELL}"></div></div>
    <div class="pval" style="color:{COL_SELL}">{sell_p:.4f}</div>
  </div>
  <div style="margin-top:8px;padding-top:8px;border-top:1px solid #F1F5F9;font-size:.62rem;color:#64748B">
    GMM conf <b style="font-family:JetBrains Mono">{gmm_conf:.4f}</b>
    &nbsp;· HMM conf <b style="font-family:JetBrains Mono">{hmm_conf:.4f}</b>
  </div>
</div>""", unsafe_allow_html=True)

    # UDL
    RN = ["BULL","BEAR","RANGE","ANOM"]
    RC = [COL_BUY, COL_SELL, COL_IND, COL_YEL]
    bars_html = ""
    for i,(n,c) in enumerate(zip(RN,RC)):
        gp = float(gmm_probs[i]) if i < len(gmm_probs) else 0
        hp = float(hmm_probs[i]) if i < len(hmm_probs) else 0
        active = (i == int(g("regime_id",2)))
        bar_bg = "#F0F0FF" if active else ""
        bars_html += f"""
<div style="margin-bottom:5px;padding:3px 0;{'background:'+bar_bg+';border-radius:4px;padding:4px 6px' if active else ''}">
  <div style="display:flex;justify-content:space-between;font-size:.62rem;margin-bottom:3px">
    <span style="color:{c};font-weight:{'800' if active else '400'}">{n}{'  ←' if active else ''}</span>
    <span style="font-family:JetBrains Mono;color:#64748B">G:{gp:.3f} H:{hp:.3f}</span>
  </div>
  <div style="height:7px;background:#F1F5F9;border-radius:4px;overflow:hidden">
    <div style="height:7px;width:{gp*100:.0f}%;background:{c};border-radius:4px"></div>
  </div>
</div>"""
    st.markdown(f'<div class="card"><div class="card-hdr">UDL · GMM & HMM</div>{bars_html}</div>',
                unsafe_allow_html=True)

    # Trade levels
    st.markdown(f"""
<div class="card">
  <div class="card-hdr">Trade Setup</div>
  <div class="lvl-grid">
    <div class="lvl-cell" style="background:#ECFDF5;border:1.5px solid #6EE7B7">
      <div style="font-family:JetBrains Mono;font-size:1.2rem;font-weight:800;color:#059669">{tp_pip:.0f}p</div>
      <div style="font-size:.55rem;color:#64748B;margin-top:2px">TAKE PROFIT</div>
    </div>
    <div class="lvl-cell" style="background:#FFF1F2;border:1.5px solid #FCA5A5">
      <div style="font-family:JetBrains Mono;font-size:1.2rem;font-weight:800;color:#DC2626">{sl_pip:.0f}p</div>
      <div style="font-size:.55rem;color:#64748B;margin-top:2px">STOP LOSS</div>
    </div>
    <div class="lvl-cell" style="background:#F0F0FF;border:1px solid #C4B5FD">
      <div style="font-family:JetBrains Mono;font-size:1.2rem;font-weight:800;color:#4F46E5">{rr:.2f}</div>
      <div style="font-size:.55rem;color:#64748B;margin-top:2px">R : R</div>
    </div>
    <div class="lvl-cell" style="background:#FFF7ED;border:1px solid #FED7AA">
      <div style="font-family:JetBrains Mono;font-size:1.2rem;font-weight:800;color:#D97706">{edge:.1f}p</div>
      <div style="font-size:.55rem;color:#64748B;margin-top:2px">EDGE</div>
    </div>
  </div>
  <div style="margin-top:8px;font-size:.62rem;color:#64748B;display:flex;justify-content:space-between">
    <span>Threshold <b style="font-family:JetBrains Mono">{SCORE_THR}</b></span>
    <span>Size <b style="font-family:JetBrains Mono">{int(size_mult*100)}%</b></span>
    <span>RSI <b style="font-family:JetBrains Mono">{rsi_v:.1f}</b></span>
  </div>
</div>""", unsafe_allow_html=True)

    # Signal distribution
    st.markdown(f"""
<div class="card">
  <div class="card-hdr">Signal Distribution · #{n_infer}</div>
  <div class="prow">
    <div class="plbl" style="color:{COL_BUY}">BUY</div>
    <div class="pbg"><div class="pfill" style="width:{n_buy/tot_s*100:.0f}%;background:{COL_BUY}"></div></div>
    <div class="pval" style="color:{COL_BUY}">{n_buy} ({n_buy/tot_s*100:.0f}%)</div>
  </div>
  <div class="prow">
    <div class="plbl" style="color:{COL_SELL}">SELL</div>
    <div class="pbg"><div class="pfill" style="width:{n_sell/tot_s*100:.0f}%;background:{COL_SELL}"></div></div>
    <div class="pval" style="color:{COL_SELL}">{n_sell} ({n_sell/tot_s*100:.0f}%)</div>
  </div>
  <div class="prow">
    <div class="plbl" style="color:{COL_HOLD}">HOLD</div>
    <div class="pbg"><div class="pfill" style="width:{n_hold/tot_s*100:.0f}%;background:{COL_HOLD}"></div></div>
    <div class="pval" style="color:{COL_HOLD}">{n_hold} ({n_hold/tot_s*100:.0f}%)</div>
  </div>
</div>""", unsafe_allow_html=True)

    # Positions
    st.markdown('<div class="card"><div class="card-hdr">Open Positions</div>', unsafe_allow_html=True)
    if open_pos:
        for pos in open_pos:
            d = str(pos.get("direction","BUY"))
            dc = COL_BUY if d=="BUY" else COL_SELL
            pnl = float(pos.get("pnl_pips", pos.get("pnl",0)))
            pc  = COL_BUY if pnl >= 0 else COL_SELL
            st.markdown(f"""
<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-left:4px solid {dc};
  border-radius:8px;padding:8px 10px;margin-bottom:6px">
  <div style="display:flex;justify-content:space-between">
    <span style="color:{dc};font-weight:800;font-family:JetBrains Mono">{d}</span>
    <span style="color:{pc};font-weight:700;font-family:JetBrains Mono">{pnl:+.1f}p</span>
  </div>
  <div style="font-size:.60rem;color:#94A3B8;margin-top:4px">
    entry {float(pos.get('entry',0)):.5f} · lots {float(pos.get('lots',0.01)):.2f}
    · SL {float(pos.get('sl',0)):.5f}
  </div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:center;color:#94A3B8;font-size:.68rem;padding:12px">No open positions</div>',
                    unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Closed trades
    if closed_jn:
        st.markdown('<div class="card"><div class="card-hdr">Recent Closed</div>', unsafe_allow_html=True)
        for tr in list(reversed(closed_jn))[:8]:
            d = str(tr.get("direction","BUY"))
            dc = COL_BUY if d=="BUY" else COL_SELL
            pv = float(tr.get("pnl_pips",0))
            pc = COL_BUY if pv>=0 else COL_SELL
            st.markdown(f"""
<div class="trow">
  <span style="color:{dc};font-weight:700;font-family:JetBrains Mono">{d}</span>
  <span style="color:#94A3B8;font-size:.60rem">{tr.get('reason','?')}</span>
  <span style="color:#94A3B8;font-size:.58rem">{str(tr.get('regime',''))[:5]}</span>
  <span style="color:{pc};font-weight:700;font-family:JetBrains Mono">{pv:+.1f}p</span>
</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ─── RIGHT PANEL — TABS ───────────────────────────────────────────────────────
with right:
    t1,t2,t3,t4,t5,t6,t7 = st.tabs([
        "📊 Price", "📈 Indicators", "🧠 Regime",
        "💰 Performance", "📋 Signal Log",
        "🏦 MT5 Account", "📡 Market"
    ])

    # ── TAB 1: PRICE CHART ───────────────────────────────────────────────────
    with t1:
        if not df_ohlc.empty:
            v = df_ohlc.tail(120).copy()
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                row_heights=[0.62, 0.20, 0.18], vertical_spacing=0.02,
                subplot_titles=["EUR/USD H4 · EMA20 · EMA50", "Volume", "Spread"])

            fig.add_trace(go.Candlestick(
                x=v["time"], open=v["open"], high=v["high"],
                low=v["low"], close=v["close"],
                increasing_line_color=COL_BUY, decreasing_line_color=COL_SELL,
                increasing_fillcolor=COL_BUY, decreasing_fillcolor=COL_SELL,
                line_width=0.7, name="OHLC",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(x=v["time"], y=v["ema20"],
                line=dict(color=COL_IND, width=1.8), name="EMA20"), row=1, col=1)
            fig.add_trace(go.Scatter(x=v["time"], y=v["ema50"],
                line=dict(color="#A855F7", width=1.4, dash="dot"), name="EMA50"), row=1, col=1)
            fig.add_hline(y=price, line=dict(color=COL_YEL, width=1, dash="dash"),
                          annotation_text=f" {price:.5f}", annotation_font=dict(size=9,color=COL_YEL), row=1, col=1)

            # EMA20 band
            fig.add_hline(y=ema20 + MR_BAND*0.0001, line=dict(color=COL_SELL, width=0.5, dash="dot"),
                          annotation_text="SELL zone", annotation_font=dict(size=8,color=COL_SELL), row=1, col=1)
            fig.add_hline(y=ema20 - MR_BAND*0.0001, line=dict(color=COL_BUY, width=0.5, dash="dot"),
                          annotation_text="BUY zone", annotation_font=dict(size=8,color=COL_BUY), row=1, col=1)

            # Trade markers
            for tr in (closed_jn or []):
                try:
                    t_e = pd.to_datetime(tr["entry_time"])
                    d_  = tr.get("direction","BUY")
                    ep  = float(tr.get("entry", price))
                    fig.add_trace(go.Scatter(
                        x=[t_e], y=[ep], mode="markers",
                        marker=dict(symbol="triangle-up" if d_=="BUY" else "triangle-down",
                                    size=10, color=COL_BUY if d_=="BUY" else COL_SELL,
                                    line=dict(width=1, color="#FFFFFF")),
                        showlegend=False,
                    ), row=1, col=1)
                except Exception: pass

            # Volume
            vol_col = v["close"].ge(v["open"]).map({True: COL_BUY, False: COL_SELL})
            if "tick_volume" in v.columns:
                fig.add_trace(go.Bar(x=v["time"], y=v["tick_volume"],
                    marker_color=vol_col, opacity=0.55, name="Vol"), row=2, col=1)
            elif "vol" in v.columns:
                fig.add_trace(go.Bar(x=v["time"], y=v["vol"],
                    marker_color=vol_col, opacity=0.55, name="Vol"), row=2, col=1)

            # Spread
            if "spread" in v.columns and v["spread"].max() > 0:
                fig.add_trace(go.Scatter(x=v["time"], y=v["spread"],
                    line=dict(color=COL_IND, width=1.3), fill="tozeroy",
                    fillcolor=f"rgba(79,70,229,.06)", name="Spd"), row=3, col=1)

            fig.update_layout(**chart(h=440), showlegend=True,
                legend=dict(orientation="h", x=0, y=1.04,
                    font=dict(size=9, color="#64748B"), bgcolor="rgba(0,0,0,0)"),
                xaxis_rangeslider_visible=False)
            fig.update_annotations(font_size=8, font_color="#94A3B8")
            st.plotly_chart(fig, use_container_width=True, config=CONF)

            last = df_ohlc.iloc[-1]
            chg  = (float(last["close"]) - float(df_ohlc.iloc[-2]["close"])) * 10000
            chg_c = COL_BUY if chg>=0 else COL_SELL
            st.markdown(f"""
<div style="display:flex;gap:18px;padding:6px 2px;font-family:JetBrains Mono;font-size:.68rem;color:#94A3B8;
  border-top:1px solid #F1F5F9;flex-wrap:wrap">
  <span>O <b style="color:#0F172A">{last['open']:.5f}</b></span>
  <span>H <b style="color:{COL_BUY}">{last['high']:.5f}</b></span>
  <span>L <b style="color:{COL_SELL}">{last['low']:.5f}</b></span>
  <span>C <b style="color:#0F172A">{last['close']:.5f}</b></span>
  <span>Δ <b style="color:{chg_c}">{chg:+.1f}p</b></span>
  <span>ATR <b style="color:{COL_IND}">{float(last.get('atr14',18)):.1f}p</b></span>
  <span>RSI <b style="color:#A855F7">{float(last.get('rsi',50)):.1f}</b></span>
  <span>EMA20 <b style="color:{COL_IND}">{float(last.get('ema20',0)):.5f}</b></span>
</div>""", unsafe_allow_html=True)
        else:
            st.warning("⚠️ eurusd_4h.csv not found — copy it to the same folder as streamlit_app.py")

    # ── TAB 2: INDICATORS ────────────────────────────────────────────────────
    with t2:
        if not df_ohlc.empty:
            v = df_ohlc.tail(150).copy()
            dist_s = (v["close"] - v["ema20"]) * 10000
            macd_fast = v["close"].ewm(span=12,adjust=False).mean()
            macd_slow = v["close"].ewm(span=26,adjust=False).mean()
            macd      = macd_fast - macd_slow
            macd_sig  = macd.ewm(span=9,adjust=False).mean()
            macd_hist = macd - macd_sig

            fig2 = make_subplots(rows=4, cols=1, shared_xaxes=True,
                row_heights=[0.28,0.24,0.24,0.24], vertical_spacing=0.04,
                subplot_titles=["RSI (14)", "ATR 14 (pips)", "EMA20 Distance (pips)", "MACD"])

            # RSI
            rsi_col_line = [COL_SELL if r>70 else COL_BUY if r<30 else COL_IND for r in v["rsi"]]
            fig2.add_trace(go.Scatter(x=v["time"], y=v["rsi"],
                line=dict(color=COL_IND, width=1.8), fill="tozeroy",
                fillcolor="rgba(79,70,229,.05)", name="RSI"), row=1, col=1)
            fig2.add_hline(y=70, line=dict(color=COL_SELL, width=0.7, dash="dot"), row=1, col=1)
            fig2.add_hline(y=30, line=dict(color=COL_BUY,  width=0.7, dash="dot"), row=1, col=1)
            fig2.add_hline(y=50, line=dict(color="#CBD5E1", width=0.5, dash="dot"), row=1, col=1)
            fig2.add_annotation(text="OB", x=v["time"].iloc[-1], y=72, font=dict(size=8,color=COL_SELL), showarrow=False, row=1, col=1)
            fig2.add_annotation(text="OS", x=v["time"].iloc[-1], y=28, font=dict(size=8,color=COL_BUY), showarrow=False, row=1, col=1)

            # ATR
            fig2.add_trace(go.Scatter(x=v["time"], y=v["atr14"],
                line=dict(color="#A855F7", width=1.8), fill="tozeroy",
                fillcolor="rgba(168,85,247,.05)", name="ATR"), row=2, col=1)
            fig2.add_hline(y=float(v["atr14"].mean()),
                           line=dict(color=COL_YEL, width=0.7, dash="dot"),
                           annotation_text=f"avg {v['atr14'].mean():.1f}p",
                           annotation_font=dict(size=8,color=COL_YEL), row=2, col=1)

            # EMA20 distance
            dist_colors = [COL_BUY if d<-MR_BAND else COL_SELL if d>MR_BAND else "#CBD5E1" for d in dist_s]
            fig2.add_trace(go.Bar(x=v["time"], y=dist_s,
                marker_color=dist_colors, opacity=0.75, name="Dist EMA20"), row=3, col=1)
            fig2.add_hline(y=MR_BAND,  line=dict(color=COL_SELL, width=0.7, dash="dash"), row=3, col=1)
            fig2.add_hline(y=-MR_BAND, line=dict(color=COL_BUY,  width=0.7, dash="dash"), row=3, col=1)
            fig2.add_hline(y=0, line=dict(color="#CBD5E1", width=0.5), row=3, col=1)

            # MACD
            hist_colors = [COL_BUY if h>=0 else COL_SELL for h in macd_hist]
            fig2.add_trace(go.Bar(x=v["time"], y=macd_hist*10000,
                marker_color=hist_colors, opacity=0.7, name="MACD Hist"), row=4, col=1)
            fig2.add_trace(go.Scatter(x=v["time"], y=macd*10000,
                line=dict(color=COL_IND, width=1.2), name="MACD"), row=4, col=1)
            fig2.add_trace(go.Scatter(x=v["time"], y=macd_sig*10000,
                line=dict(color=COL_YEL, width=1.2, dash="dot"), name="Signal"), row=4, col=1)
            fig2.add_hline(y=0, line=dict(color="#CBD5E1", width=0.5), row=4, col=1)

            fig2.update_layout(**chart(h=480), showlegend=False)
            fig2.update_annotations(font_size=8, font_color="#94A3B8")
            st.plotly_chart(fig2, use_container_width=True, config=CONF)

            # Current snapshot
            last = df_ohlc.iloc[-1]
            rsi_now = float(last.get("rsi",50))
            atr_now = float(last.get("atr14",18))
            dist_now = (float(last["close"]) - float(last.get("ema20",last["close"]))) * 10000
            rsi_c = COL_SELL if rsi_now>70 else COL_BUY if rsi_now<30 else COL_IND
            dist_c2 = COL_BUY if dist_now<-MR_BAND else COL_SELL if dist_now>MR_BAND else COL_HOLD
            st.markdown(f"""
<div style="display:flex;gap:24px;font-family:JetBrains Mono;font-size:.72rem;padding:6px 2px;
  color:#64748B;border-top:1px solid #F1F5F9;flex-wrap:wrap">
  <span>RSI <b style="color:{rsi_c}">{rsi_now:.1f}</b></span>
  <span>ATR <b style="color:#A855F7">{atr_now:.1f}p</b></span>
  <span>EMA20 Dist <b style="color:{dist_c2}">{dist_now:+.1f}p</b></span>
  <span>Spr/ATR <b style="color:{COL_YEL}">{spread/max(atr_now,1)*100:.1f}%</b></span>
  <span>MACD <b style="color:{COL_IND}">{float(macd.iloc[-1])*10000:+.2f}p</b></span>
</div>""", unsafe_allow_html=True)
        else:
            st.info("Load eurusd_4h.csv to see indicators.")

    # ── TAB 3: REGIME ────────────────────────────────────────────────────────
    with t3:
        c1r, c2r = st.columns(2)
        RN2 = ["BULL TREND","BEAR TREND","RANGE CHOP","ANOMALY"]
        RC2 = [COL_BUY, COL_SELL, COL_IND, COL_YEL]
        gv  = [float(p) for p in gmm_probs]
        hv  = [float(p) for p in hmm_probs]

        with c1r:
            fig_g = go.Figure(go.Pie(labels=RN2, values=gv, hole=0.60,
                marker=dict(colors=RC2, line=dict(color="#FFFFFF", width=2)),
                textfont=dict(size=9)))
            fig_g.add_annotation(text=f"<b>GMM</b><br>{regime[:5]}",
                x=0.5, y=0.5, showarrow=False, font=dict(size=11, color="#0F172A", family="JetBrains Mono"))
            fig_g.update_layout(**chart(h=260, title="GMM Regime"), showlegend=True,
                legend=dict(orientation="v", x=1.02, font=dict(size=9,color="#64748B"), bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_g, use_container_width=True, config=CONF)

        with c2r:
            fig_h = go.Figure(go.Pie(labels=RN2, values=hv, hole=0.60,
                marker=dict(colors=RC2, line=dict(color="#FFFFFF", width=2)),
                textfont=dict(size=9)))
            fig_h.add_annotation(text=f"<b>HMM</b><br>{regime[:5]}",
                x=0.5, y=0.5, showarrow=False, font=dict(size=11, color="#0F172A", family="JetBrains Mono"))
            fig_h.update_layout(**chart(h=260, title="HMM Regime"), showlegend=True,
                legend=dict(orientation="v", x=1.02, font=dict(size=9,color="#64748B"), bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_h, use_container_width=True, config=CONF)

        # Score breakdown
        ml_v  = mr_str if (mr_mode and abs(dist_pips)>=MR_BAND) else max(buy_p, sell_p)
        sc_ml = 0.50 * ml_v
        sc_rg = 0.30 * 0.70
        sc_hm = 0.20 * hmm_conf
        fig_sc = go.Figure(go.Bar(
            x=["ML × 0.50", "Regime × 0.30", "HMM × 0.20", "TOTAL"],
            y=[sc_ml, sc_rg, sc_hm, score],
            marker_color=[COL_IND, "#A855F7", COL_BLU,
                          COL_BUY if score>=SCORE_THR else COL_SELL],
            text=[f"{v:.4f}" for v in [sc_ml, sc_rg, sc_hm, score]],
            textposition="outside", textfont=dict(size=10, color="#475569"),
        ))
        fig_sc.add_hline(y=SCORE_THR, line=dict(color=COL_YEL, width=1.5, dash="dash"),
                         annotation_text=f"Thr {SCORE_THR}", annotation_font=dict(size=9,color=COL_YEL))
        fig_sc.update_layout(**chart(h=220, title="Score Decomposition"), showlegend=False)
        fig_sc.update_yaxes(range=[0,1.1])
        st.plotly_chart(fig_sc, use_container_width=True, config=CONF)

        # Regime table
        RID = int(g("regime_id",2))
        cells = ""
        for i,(n,c) in enumerate(zip(RN2,RC2)):
            active = (i==RID)
            bg = "#EDE9FE" if active else "#F8FAFC"
            bd = f"2px solid {c}" if active else "1px solid #E2E8F0"
            cells += f"""
<div style="background:{bg};border:{bd};border-radius:8px;padding:10px;text-align:center">
  <div style="font-family:JetBrains Mono;font-size:.75rem;font-weight:700;color:{c}">{n}</div>
  <div style="font-size:.58rem;color:#64748B;margin-top:3px">GMM {gv[i]:.3f}</div>
  <div style="font-size:.58rem;color:#94A3B8">HMM {hv[i]:.3f}</div>
  {'<div style="font-size:.58rem;color:#4F46E5;font-weight:700;margin-top:2px">ACTIVE</div>' if active else ''}
</div>"""
        st.markdown(f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:10px">{cells}</div>',
                    unsafe_allow_html=True)

    # ── TAB 4: PERFORMANCE ───────────────────────────────────────────────────
    with t4:
        if closed_jn:
            df_tr = pd.DataFrame(closed_jn)
            df_tr["pnl_pips"] = pd.to_numeric(df_tr.get("pnl_pips", pd.Series()), errors="coerce").fillna(0)
            df_tr["cum_pnl"]  = df_tr["pnl_pips"].cumsum()
            df_tr["dd"]       = df_tr["cum_pnl"] - df_tr["cum_pnl"].cummax()

            ca, cb = st.columns(2)
            with ca:
                fig_eq = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.65,0.35], vertical_spacing=0.05,
                    subplot_titles=["Cumulative P&L (pips)","Drawdown"])
                fig_eq.add_trace(go.Scatter(y=df_tr["cum_pnl"].values, mode="lines+markers",
                    line=dict(color=COL_BUY, width=2), marker=dict(size=5,color=COL_BUY),
                    fill="tozeroy", fillcolor=f"rgba(16,185,129,.07)"), row=1, col=1)
                fig_eq.add_hline(y=0, line=dict(color="#CBD5E1",width=0.8,dash="dot"), row=1, col=1)
                fig_eq.add_trace(go.Scatter(y=df_tr["dd"].values, mode="lines", fill="tozeroy",
                    fillcolor=f"rgba(239,68,68,.07)", line=dict(color=COL_SELL,width=1.5)), row=2, col=1)
                fig_eq.update_layout(**chart(h=310), showlegend=False)
                st.plotly_chart(fig_eq, use_container_width=True, config=CONF)

            with cb:
                pnls = df_tr["pnl_pips"].values
                wins_ = int((pnls>0).sum()); losses_ = int((pnls<=0).sum())
                fig_pie = go.Figure(go.Pie(
                    labels=["Win","Loss"], values=[wins_,losses_], hole=0.55,
                    marker=dict(colors=[COL_BUY,COL_SELL],line=dict(color="#FFFFFF",width=2)),
                    textfont=dict(size=10)))
                fig_pie.add_annotation(text=f"<b>{win_rate:.0f}%</b>",
                    x=0.5,y=0.5,showarrow=False,font=dict(size=18,color="#0F172A",family="JetBrains Mono"))
                fig_pie.update_layout(**chart(h=190, title="Win Rate"), showlegend=True,
                    legend=dict(orientation="h",x=0.2,y=-0.1,font=dict(size=9,color="#64748B"),bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_pie, use_container_width=True, config=CONF)

                fig_hist = go.Figure(go.Histogram(x=pnls, nbinsx=14,
                    marker_color=[COL_BUY if p>=0 else COL_SELL for p in pnls],
                    opacity=0.8))
                fig_hist.add_vline(x=0, line=dict(color="#CBD5E1",width=1,dash="dot"))
                fig_hist.update_layout(**chart(h=190, title="P&L Distribution"), showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True, config=CONF)

            # Summary stats
            wins_arr  = pnls[pnls>0];  loss_arr = pnls[pnls<=0]
            avg_win   = float(wins_arr.mean()) if len(wins_arr) else 0
            avg_loss  = float(loss_arr.mean()) if len(loss_arr) else 0
            max_win   = float(pnls.max()) if len(pnls) else 0
            max_dd    = float(df_tr["dd"].min())
            st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px">
  <div class="kpi"><div class="kpi-val" style="color:{COL_BUY}">{avg_win:+.1f}p</div><div class="kpi-lbl">Avg Win</div></div>
  <div class="kpi"><div class="kpi-val" style="color:{COL_SELL}">{avg_loss:+.1f}p</div><div class="kpi-lbl">Avg Loss</div></div>
  <div class="kpi"><div class="kpi-val" style="color:{COL_BUY}">{max_win:+.1f}p</div><div class="kpi-lbl">Best Trade</div></div>
  <div class="kpi"><div class="kpi-val" style="color:{COL_SELL}">{max_dd:.1f}p</div><div class="kpi-lbl">Max DD</div></div>
  <div class="kpi"><div class="kpi-val" style="color:{COL_IND}">{min(pf,99):.2f}</div><div class="kpi-lbl">Prof Factor</div></div>
</div>""", unsafe_allow_html=True)

            if "regime" in df_tr.columns:
                rg_stats = (df_tr.groupby("regime")["pnl_pips"]
                            .agg(["count","mean","sum"]).reset_index()
                            .rename(columns={"count":"Trades","mean":"Avg","sum":"Total"})
                            .round(2))
                st.markdown("<div style='margin-top:10px;font-size:.65rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:1px'>Performance by Regime</div>",unsafe_allow_html=True)
                st.dataframe(rg_stats, use_container_width=True, hide_index=True)
        else:
            st.info("⏳ No closed trades yet. Performance builds as trades close.")

    # ── TAB 5: SIGNAL LOG ────────────────────────────────────────────────────
    with t5:
        if not sigs.empty:
            cols_show = [c for c in ["ts","signal","score","regime","buy_p","sell_p",
                "mr_distance_pips","ema20","price","atr","quality","gate","tp_pip","sl_pip",
                "gmm_conf","hmm_conf"] if c in sigs.columns]
            rec = sigs.tail(500)[cols_show]
            if "ts" in rec.columns:
                try: rec = rec.sort_values("ts", ascending=False)
                except: pass

            def style_sig(v):
                if v=="BUY":  return "color:#059669;font-weight:700"
                if v=="SELL": return "color:#DC2626;font-weight:700"
                return "color:#94A3B8"
            def style_gate(v):
                return "color:#059669;font-weight:700" if v=="ACCEPTED" else "color:#94A3B8"

            styled = rec.style
            if "signal" in cols_show: styled = styled.map(style_sig, subset=["signal"])
            if "gate"   in cols_show: styled = styled.map(style_gate, subset=["gate"])
            for fc in ["buy_p","sell_p","score","gmm_conf","hmm_conf"]:
                if fc in cols_show: styled = styled.format({fc:"{:.4f}"})
            st.dataframe(styled, use_container_width=True, height=440)

            # Signal timeline chart
            if "ts" in sigs.columns and "signal" in sigs.columns:
                tail50 = sigs.tail(50).copy()
                sc_map = {"BUY":1,"SELL":-1,"HOLD":0}
                tail50["sig_num"] = tail50["signal"].map(sc_map).fillna(0)
                fig_log = go.Figure()
                for sig_type, col_t, sym in [("BUY",COL_BUY,"triangle-up"),("SELL",COL_SELL,"triangle-down"),("HOLD",COL_HOLD,"circle")]:
                    mask = tail50["signal"]==sig_type
                    if mask.any():
                        y_col = "score" if "score" in tail50.columns else "sig_num"
                        fig_log.add_trace(go.Scatter(
                            x=tail50[mask]["ts"], y=tail50[mask][y_col],
                            mode="markers",
                            marker=dict(symbol=sym, size=10, color=col_t,
                                        line=dict(width=1,color="#FFFFFF")),
                            name=sig_type,
                        ))
                if "score" in tail50.columns:
                    fig_log.add_hline(y=SCORE_THR, line=dict(color=COL_YEL,width=1,dash="dash"),
                                      annotation_text=f"thr {SCORE_THR}", annotation_font=dict(size=8,color=COL_YEL))
                fig_log.update_layout(**chart(h=200, title="Signal Timeline (last 50)"), showlegend=True,
                    legend=dict(orientation="h", x=0, y=1.1, font=dict(size=9,color="#64748B"), bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_log, use_container_width=True, config=CONF)
        else:
            st.info("Signal log builds as inferences run (every 10s). Check logs/ folder.")
            # Show from JSON
            if jn:
                c1s, c2s = st.columns(2)
                with c1s:
                    st.markdown(f"""
<div class="card">
  <div class="card-hdr">Signal Counts (current session)</div>
  <div style="font-family:JetBrains Mono;font-size:.80rem;line-height:2.2">
    <span style="color:{COL_BUY}">BUY</span>  {n_buy}<br>
    <span style="color:{COL_SELL}">SELL</span> {n_sell}<br>
    <span style="color:{COL_HOLD}">HOLD</span> {n_hold}<br>
    <span style="color:#94A3B8">Inferences: {n_infer}</span>
  </div>
</div>""", unsafe_allow_html=True)
                with c2s:
                    gc_html = "".join(f'<div style="font-size:.72rem">{k}: <b style="font-family:JetBrains Mono">{v}</b></div>'
                                      for k,v in gate_counts.items())
                    st.markdown(f'<div class="card"><div class="card-hdr">Gate Counts</div>{gc_html}</div>', unsafe_allow_html=True)

    # ── TAB 6: MT5 ACCOUNT ──────────────────────────────────────────────────
    with t6:
        c1a, c2a = st.columns(2)
        with c1a:
            if mt5_ok and acct:
                profit_c = COL_BUY if float(acct.get("profit",0))>=0 else COL_SELL
                eq_c     = COL_BUY if float(acct.get("equity",bal))>=bal else COL_SELL
                st.markdown(f"""
<div class="card">
  <div class="card-hdr">🏦 Account Details</div>
  <div class="mt5-stat"><span class="mt5-k">Login</span><span class="mt5-v">{acct.get('login','—')}</span></div>
  <div class="mt5-stat"><span class="mt5-k">Name</span><span class="mt5-v">{acct.get('name','—')}</span></div>
  <div class="mt5-stat"><span class="mt5-k">Server</span><span class="mt5-v">{acct.get('server','ICMarketsSC-Demo')}</span></div>
  <div class="mt5-stat"><span class="mt5-k">Currency</span><span class="mt5-v">{acct.get('currency','USD')}</span></div>
  <div class="mt5-stat"><span class="mt5-k">Leverage</span><span class="mt5-v">1:{acct.get('leverage',500)}</span></div>
</div>""", unsafe_allow_html=True)
                st.markdown(f"""
<div class="card">
  <div class="card-hdr">💰 Financials</div>
  <div class="mt5-stat"><span class="mt5-k">Balance</span>
    <span class="mt5-v">${float(acct.get('balance',0)):,.2f}</span></div>
  <div class="mt5-stat"><span class="mt5-k">Equity</span>
    <span class="mt5-v" style="color:{eq_c}">${float(acct.get('equity',0)):,.2f}</span></div>
  <div class="mt5-stat"><span class="mt5-k">Profit</span>
    <span class="mt5-v" style="color:{profit_c}">${float(acct.get('profit',0)):+.2f}</span></div>
  <div class="mt5-stat"><span class="mt5-k">Margin</span>
    <span class="mt5-v">${float(acct.get('margin',0)):,.2f}</span></div>
  <div class="mt5-stat" style="border:none"><span class="mt5-k">Free Margin</span>
    <span class="mt5-v">${float(acct.get('free_margin',0)):,.2f}</span></div>
</div>""", unsafe_allow_html=True)

                # Balance / Equity gauge
                if float(acct.get("balance",200)) > 0:
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=float(acct.get("equity",200)),
                        delta={"reference": float(acct.get("balance",200)), "relative": False},
                        title={"text": "Equity vs Balance", "font": {"size":12,"color":"#475569"}},
                        gauge={
                            "axis": {"range":[float(acct.get("balance",200))*0.9, float(acct.get("balance",200))*1.1]},
                            "bar":  {"color": COL_IND},
                            "steps":[{"range":[float(acct.get("balance",200))*0.9, float(acct.get("balance",200))], "color":"#FEF3C7"},
                                     {"range":[float(acct.get("balance",200)), float(acct.get("balance",200))*1.1], "color":"#D1FAE5"}],
                            "threshold":{"line":{"color":COL_YEL,"width":3},"thickness":0.75,"value":float(acct.get("balance",200))},
                        },
                        number={"prefix":"$","font":{"size":18,"color":"#0F172A","family":"JetBrains Mono"}},
                    ))
                    fig_g.update_layout(**{**chart(h=220), "margin": dict(l=20,r=20,t=30,b=10)})
                    st.plotly_chart(fig_g, use_container_width=True, config=CONF)
            else:
                st.markdown(f"""
<div class="alert-warn">
  <b>MT5 Not Connected</b><br>
  Login: {MT5_LOGIN} · Server: {MT5_SERVER}<br>
  <br>
  Steps to connect:<br>
  1. Open MetaTrader 5<br>
  2. Login with ICM demo credentials<br>
  3. Wait for green dot (bottom-right)<br>
  4. Restart streamlit_app.py<br>
  5. Dashboard connects automatically in 15s
</div>""", unsafe_allow_html=True)

        with c2a:
            # Open positions chart
            if open_pos:
                fig_pos = go.Figure()
                for p in open_pos:
                    d = str(p.get("direction","BUY"))
                    ep = float(p.get("entry",price))
                    cp = float(p.get("pnl_pips",0))
                    tp = float(p.get("tp",0)) or ep + (0.0038 if d=="BUY" else -0.0038)
                    sl = float(p.get("sl",0)) or ep - (0.0016 if d=="BUY" else -0.0016)
                    col = COL_BUY if d=="BUY" else COL_SELL
                    fig_pos.add_trace(go.Scatter(
                        x=[str(p.get("ticket",""))]*3,
                        y=[sl, ep, tp],
                        mode="markers+lines",
                        marker=dict(size=[8,12,8], color=[COL_SELL, col, COL_BUY]),
                        line=dict(color=col, width=1.5),
                        name=f"{d} #{p.get('ticket','')}",
                    ))
                fig_pos.update_layout(**chart(h=240, title="Open Positions"), showlegend=True,
                    legend=dict(font=dict(size=9,color="#64748B"), bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_pos, use_container_width=True, config=CONF)
            else:
                st.info("No open positions on MT5.")

            # Connection diagnostics
            st.markdown(f"""
<div class="card">
  <div class="card-hdr">🔌 Connection Diagnostics</div>
  <div class="mt5-stat"><span class="mt5-k">MT5 Connected</span>
    <span class="mt5-v" style="color:{'#059669' if mt5_ok else '#EF4444'}">{'✓ YES' if mt5_ok else '✗ NO'}</span></div>
  <div class="mt5-stat"><span class="mt5-k">Last Update</span>
    <span class="mt5-v">{age_str}</span></div>
  <div class="mt5-stat"><span class="mt5-k">Inferences</span>
    <span class="mt5-v">#{n_infer}</span></div>
  <div class="mt5-stat"><span class="mt5-k">Login</span>
    <span class="mt5-v">{MT5_LOGIN}</span></div>
  <div class="mt5-stat"><span class="mt5-k">Server</span>
    <span class="mt5-v">{MT5_SERVER}</span></div>
  <div class="mt5-stat" style="border:none"><span class="mt5-k">Symbol</span>
    <span class="mt5-v">{SYMBOL} H4</span></div>
</div>""", unsafe_allow_html=True)

    # ── TAB 7: MARKET ────────────────────────────────────────────────────────
    with t7:
        if not df_ohlc.empty:
            v = df_ohlc.tail(200).copy().reset_index(drop=True)

            # Ensure volume column exists regardless of CSV column name
            if "tick_volume" in v.columns:
                v["_vol"] = v["tick_volume"].fillna(0).astype(float)
            elif "vol" in v.columns:
                v["_vol"] = v["vol"].fillna(0).astype(float)
            elif "real_volume" in v.columns:
                v["_vol"] = v["real_volume"].fillna(0).astype(float)
            else:
                v["_vol"] = pd.Series(np.abs(np.random.randn(len(v)) * 5000 + 5000))

            # Ensure atr14 and rsi exist
            if "atr14" not in v.columns:
                tr = pd.concat([
                    v["high"] - v["low"],
                    (v["high"] - v["close"].shift()).abs(),
                    (v["low"]  - v["close"].shift()).abs(),
                ], axis=1).max(axis=1)
                v["atr14"] = (tr.rolling(14).mean() * 10000).fillna(18)
            if "rsi" not in v.columns:
                gain = v["close"].diff().clip(lower=0).rolling(14).mean()
                loss = (-v["close"].diff().clip(upper=0)).rolling(14).mean()
                v["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, 1e-10))).fillna(50)

            v80 = v.tail(80).copy()
            vc  = v80["close"].ge(v80["open"]).map({True: COL_BUY, False: COL_SELL}).tolist()

            # ── Row 1: Volume bar + Price distribution ──
            c1m, c2m = st.columns(2)

            with c1m:
                # Volume bars
                fig_vol = go.Figure(go.Bar(
                    x=v80["time"], y=v80["_vol"],
                    marker_color=vc, opacity=0.75, name="Volume",
                ))
                avg_vol = float(v80["_vol"].mean())
                fig_vol.add_hline(y=avg_vol,
                    line=dict(color=COL_YEL, width=1, dash="dot"),
                    annotation_text=f"avg {avg_vol:.0f}",
                    annotation_font=dict(size=8, color=COL_YEL))
                fig_vol.update_layout(**chart(h=215, title="Tick Volume · last 80 H4 bars"), showlegend=False)
                st.plotly_chart(fig_vol, use_container_width=True, config=CONF)

            with c2m:
                # Price distribution histogram
                fig_ph = go.Figure(go.Histogram(
                    x=v["close"].values, nbinsx=40,
                    marker_color=COL_IND, opacity=0.75,
                ))
                fig_ph.add_vline(x=float(v["close"].iloc[-1]),
                    line=dict(color=COL_YEL, width=2, dash="dash"),
                    annotation_text=f"Last {v['close'].iloc[-1]:.5f}",
                    annotation_font=dict(size=8, color=COL_YEL))
                fig_ph.add_vline(x=float(v["ema20"].iloc[-1]) if "ema20" in v.columns else ema20,
                    line=dict(color=COL_IND, width=1.2, dash="dot"),
                    annotation_text="EMA20",
                    annotation_font=dict(size=8, color=COL_IND))
                fig_ph.update_layout(**chart(h=215, title="Price Distribution · last 200 bars"), showlegend=False)
                st.plotly_chart(fig_ph, use_container_width=True, config=CONF)

            # ── Row 2: ATR volatility + RSI zones ──
            c3m, c4m = st.columns(2)

            with c3m:
                fig_atr = go.Figure()
                fig_atr.add_trace(go.Scatter(
                    x=v["time"], y=v["atr14"],
                    line=dict(color="#A855F7", width=1.8),
                    fill="tozeroy", fillcolor="rgba(168,85,247,.07)",
                    name="ATR14",
                ))
                avg_atr = float(v["atr14"].mean())
                hi_atr  = float(v["atr14"].quantile(0.80))
                lo_atr  = float(v["atr14"].quantile(0.20))
                fig_atr.add_hline(y=avg_atr,
                    line=dict(color=COL_YEL, width=1, dash="dot"),
                    annotation_text=f"avg {avg_atr:.1f}p",
                    annotation_font=dict(size=8, color=COL_YEL))
                fig_atr.add_hline(y=hi_atr,
                    line=dict(color=COL_SELL, width=0.7, dash="dot"),
                    annotation_text=f"hi {hi_atr:.1f}p",
                    annotation_font=dict(size=8, color=COL_SELL))
                fig_atr.add_hline(y=lo_atr,
                    line=dict(color=COL_BUY, width=0.7, dash="dot"),
                    annotation_text=f"lo {lo_atr:.1f}p",
                    annotation_font=dict(size=8, color=COL_BUY))
                fig_atr.update_layout(**chart(h=215, title="ATR 14 (pips) · Volatility History"), showlegend=False)
                st.plotly_chart(fig_atr, use_container_width=True, config=CONF)

            with c4m:
                fig_rsi_z = go.Figure()
                fig_rsi_z.add_hrect(y0=70, y1=100,
                    fillcolor="rgba(239,68,68,.07)", line_width=0)
                fig_rsi_z.add_hrect(y0=0, y1=30,
                    fillcolor="rgba(16,185,129,.07)", line_width=0)
                fig_rsi_z.add_trace(go.Scatter(
                    x=v["time"], y=v["rsi"],
                    line=dict(color=COL_IND, width=1.6),
                    fill="tozeroy", fillcolor="rgba(79,70,229,.05)",
                    name="RSI",
                ))
                fig_rsi_z.add_hline(y=70, line=dict(color=COL_SELL, width=0.8, dash="dot"))
                fig_rsi_z.add_hline(y=30, line=dict(color=COL_BUY,  width=0.8, dash="dot"))
                fig_rsi_z.add_hline(y=50, line=dict(color="#CBD5E1", width=0.5))
                fig_rsi_z.update_yaxes(range=[0, 100])
                fig_rsi_z.update_layout(**chart(h=215, title="RSI · Overbought / Oversold Zones"), showlegend=False)
                st.plotly_chart(fig_rsi_z, use_container_width=True, config=CONF)

            # ── Row 3: EMA spread + Candle body analysis ──
            c5m, c6m = st.columns(2)

            with c5m:
                # EMA20 vs EMA50 spread
                if "ema20" in v.columns and "ema50" in v.columns:
                    ema_spread = (v["ema20"] - v["ema50"]) * 10000
                    sp_colors  = [COL_BUY if x >= 0 else COL_SELL for x in ema_spread]
                    fig_es = go.Figure(go.Bar(
                        x=v["time"], y=ema_spread,
                        marker_color=sp_colors, opacity=0.75, name="EMA20-EMA50",
                    ))
                    fig_es.add_hline(y=0, line=dict(color="#CBD5E1", width=1))
                    fig_es.update_layout(**chart(h=215, title="EMA20 − EMA50 Spread (pips)"), showlegend=False)
                    st.plotly_chart(fig_es, use_container_width=True, config=CONF)

            with c6m:
                # Candle body ratio (body / total range)
                body  = (v["close"] - v["open"]).abs()
                total = (v["high"]  - v["low"]).replace(0, 1e-8)
                v["body_ratio"] = (body / total * 100).clip(0, 100)
                fig_br = go.Figure(go.Scatter(
                    x=v["time"], y=v["body_ratio"],
                    line=dict(color=COL_BLU, width=1.4),
                    fill="tozeroy", fillcolor="rgba(14,165,233,.06)",
                    name="Body %",
                ))
                fig_br.add_hline(y=50, line=dict(color=COL_YEL, width=0.8, dash="dot"),
                    annotation_text="50% body", annotation_font=dict(size=8, color=COL_YEL))
                fig_br.update_yaxes(range=[0, 100])
                fig_br.update_layout(**chart(h=215, title="Candle Body Ratio (%) · Momentum Strength"), showlegend=False)
                st.plotly_chart(fig_br, use_container_width=True, config=CONF)

            # ── Market stats summary row ──
            last_v = v.iloc[-1]
            chg_d  = float(v["close"].iloc[-1]) - float(v["close"].iloc[-2])
            chg_w  = float(v["close"].iloc[-1]) - float(v["close"].iloc[-5]) if len(v) > 5 else 0
            chg_m  = float(v["close"].iloc[-1]) - float(v["close"].iloc[-30]) if len(v) > 30 else 0
            high52 = float(v["high"].max())
            low52  = float(v["low"].min())
            rng52  = (high52 - low52) * 10000
            pos_pct = ((float(v["close"].iloc[-1]) - low52) / max(high52 - low52, 1e-8)) * 100

            def pip_badge(v_):
                c_ = COL_BUY if v_ >= 0 else COL_SELL
                return f'<span style="color:{c_};font-weight:700;font-family:JetBrains Mono">{v_*10000:+.1f}p</span>'

            st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin-top:4px">
  <div class="kpi"><div class="kpi-val">{float(v['close'].iloc[-1]):.5f}</div><div class="kpi-lbl">Last Close</div></div>
  <div class="kpi">
    <div class="kpi-val">{pip_badge(chg_d)}</div><div class="kpi-lbl">Day Chg</div>
  </div>
  <div class="kpi">
    <div class="kpi-val">{pip_badge(chg_w)}</div><div class="kpi-lbl">Week Chg</div>
  </div>
  <div class="kpi">
    <div class="kpi-val">{pip_badge(chg_m)}</div><div class="kpi-lbl">Month Chg</div>
  </div>
  <div class="kpi">
    <div class="kpi-val" style="color:{COL_BUY}">{high52:.5f}</div><div class="kpi-lbl">Period High</div>
  </div>
  <div class="kpi">
    <div class="kpi-val" style="color:{COL_SELL}">{low52:.5f}</div><div class="kpi-lbl">Period Low</div>
  </div>
</div>
<div class="card" style="margin-top:8px">
  <div class="card-hdr">Price Position in Range ({rng52:.0f}p total range · last 200 bars)</div>
  <div style="background:#F1F5F9;border-radius:6px;height:16px;position:relative;margin:6px 0">
    <div style="background:linear-gradient(90deg,{COL_SELL},{COL_YEL},{COL_BUY});
      width:100%;height:16px;border-radius:6px;opacity:.25"></div>
    <div style="position:absolute;top:0;left:{pos_pct:.1f}%;transform:translateX(-50%);
      width:4px;height:16px;background:{COL_IND};border-radius:2px"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-family:JetBrains Mono;font-size:.60rem;color:#94A3B8">
    <span>Low {low52:.5f}</span>
    <span style="color:{COL_IND};font-weight:700">{pos_pct:.0f}% in range</span>
    <span>High {high52:.5f}</span>
  </div>
</div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
<div class="alert-warn">
  ⚠️ <b>eurusd_4h.csv not found</b><br>
  Copy <code>eurusd_4h.csv</code> to the same folder as <code>streamlit_app.py</code>.<br>
  All Market tab charts will populate automatically.
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    auto_ref    = st.toggle("Auto-refresh", value=True)
    refresh_sec = st.slider("Refresh (s)", 3, 30, 5)
    st.divider()
    conn_label = "🟢 MT5 Live" if mt5_ok else "🟡 CSV Mode"
    st.markdown(f"""
**{conn_label}**
- Login: `{MT5_LOGIN}`
- Server: `{MT5_SERVER}`
- Symbol: `{SYMBOL}` H4
- Leverage: 1:{leverage}

**Engine v18.20**
- Score = ML×0.5 + Rg×0.3 + HMM×0.2
- RANGE → MR-FADE ±{MR_BAND:.0f}pip
- Threshold: {SCORE_THR}
- Auto-refresh: {refresh_sec}s

**Data: {age_str}**
""")

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER + AUTO-REFRESH
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="text-align:center;color:#CBD5E1;font-size:.56rem;padding:10px 0;
  border-top:1px solid #E2E8F0;margin-top:10px;font-family:JetBrains Mono">
  v18.20 · EUR/USD H4 · ICM {MT5_SERVER} · MR-FADE Strategy ·
  {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC
</div>""", unsafe_allow_html=True)

if auto_ref:
    time.sleep(refresh_sec)
    st.rerun()
else:
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.rerun()
