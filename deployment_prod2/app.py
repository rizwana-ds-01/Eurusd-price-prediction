"""
app.py  --  EUR/USD H4  v18.14  Streamlit Live Dashboard
=========================================================
White professional theme | Auto-refresh every 10 s
Real ensemble: CatBoost + LightGBM + XGBoost
Technical regime detection: ADX + EMA alignment + ATR
All signals: BUY / SELL / HOLD  |  Chart with SL/TP lines
"""

import sys
import time
import warnings
import traceback
import joblib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

# ─── PATHS ────────────────────────────────────────────────────
THIS_FILE  = Path(__file__).resolve()
ROOT       = THIS_FILE.parent
MODELS_DIR = ROOT / "models"
DATA_DIR   = ROOT / "data"
LOGS_DIR   = ROOT / "logs"
sys.path.insert(0, str(ROOT / "app"))
LOGS_DIR.mkdir(exist_ok=True)

# ─── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="EUR/USD H4 – AI Dashboard v18.14",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── GLOBAL CSS ───────────────────────────────────────────────
st.markdown("""
<style>
html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"]{
  background:#f0f4f8 !important;
  font-family:'Segoe UI',system-ui,sans-serif;
}
[data-testid="stHeader"]{background:#ffffff !important;border-bottom:1px solid #e2e8f0;}
.block-container{padding:1rem 1.2rem !important;max-width:100% !important;}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none !important;}

/* metrics */
[data-testid="metric-container"]{
  background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
  padding:12px 16px;box-shadow:0 1px 3px rgba(0,0,0,.06);
}
[data-testid="stMetricValue"]{font-size:20px !important;font-weight:800;color:#1e293b !important;}
[data-testid="stMetricLabel"]{font-size:10px !important;color:#64748b !important;
  text-transform:uppercase;letter-spacing:.8px;}
[data-testid="stMetricDelta"]{font-size:12px !important;}

/* signal badges */
.badge-BUY{background:linear-gradient(135deg,#d1fae5,#a7f3d0);border:2px solid #10b981;
  color:#065f46;font-size:40px;font-weight:900;letter-spacing:5px;
  text-align:center;border-radius:16px;padding:18px 0;
  box-shadow:0 4px 12px rgba(16,185,129,.2);}
.badge-SELL{background:linear-gradient(135deg,#fee2e2,#fecaca);border:2px solid #ef4444;
  color:#7f1d1d;font-size:40px;font-weight:900;letter-spacing:5px;
  text-align:center;border-radius:16px;padding:18px 0;
  box-shadow:0 4px 12px rgba(239,68,68,.2);}
.badge-HOLD{background:linear-gradient(135deg,#f1f5f9,#e2e8f0);border:2px solid #94a3b8;
  color:#475569;font-size:40px;font-weight:900;letter-spacing:5px;
  text-align:center;border-radius:16px;padding:18px 0;}

/* cards */
.card{background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;
  padding:16px 18px;margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,.05);}
.card-title{font-size:10px;font-weight:700;text-transform:uppercase;
  letter-spacing:1.2px;color:#94a3b8;margin-bottom:12px;}

/* prob bars */
.pb-wrap{margin-bottom:9px;}
.pb-row{display:flex;justify-content:space-between;font-size:12px;
  font-weight:600;margin-bottom:3px;}
.pb-bg{background:#f1f5f9;border-radius:6px;height:11px;overflow:hidden;}
.pb-g{height:11px;border-radius:6px;background:linear-gradient(90deg,#10b981,#34d399);}
.pb-r{height:11px;border-radius:6px;background:linear-gradient(90deg,#ef4444,#f87171);}
.pb-b{height:11px;border-radius:6px;background:linear-gradient(90deg,#3b82f6,#60a5fa);}
.pb-o{height:11px;border-radius:6px;background:linear-gradient(90deg,#f59e0b,#fbbf24);}
.pb-p{height:11px;border-radius:6px;background:linear-gradient(90deg,#8b5cf6,#a78bfa);}
.pb-gray{height:11px;border-radius:6px;background:linear-gradient(90deg,#64748b,#94a3b8);}

/* regime pill */
.pill-BULL{background:#d1fae5;border:1px solid #6ee7b7;color:#065f46;}
.pill-BEAR{background:#fee2e2;border:1px solid #fca5a5;color:#7f1d1d;}
.pill-RANGE{background:#fef3c7;border:1px solid #fcd34d;color:#78350f;}
.pill-ANOM{background:#ede9fe;border:1px solid #c4b5fd;color:#4c1d95;}
.pill{display:block;text-align:center;padding:8px 0;border-radius:20px;
  font-weight:700;font-size:15px;letter-spacing:.5px;width:100%;}

/* status */
.dot-live{display:inline-block;width:9px;height:9px;border-radius:50%;
  background:#10b981;box-shadow:0 0 7px #10b981;animation:blink 2s infinite;}
.dot-sim{display:inline-block;width:9px;height:9px;border-radius:50%;background:#f59e0b;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.live-pill{display:inline-flex;align-items:center;gap:7px;background:#d1fae5;
  color:#065f46;border:1px solid #6ee7b7;border-radius:20px;
  padding:5px 14px;font-size:12px;font-weight:600;}
.sim-pill{display:inline-flex;align-items:center;gap:7px;background:#fef3c7;
  color:#78350f;border:1px solid #fcd34d;border-radius:20px;
  padding:5px 14px;font-size:12px;font-weight:600;}

/* kpi boxes */
.kpi{background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
  padding:12px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.05);}
.kpi-lbl{font-size:10px;text-transform:uppercase;letter-spacing:.8px;
  color:#94a3b8;margin-bottom:3px;}
.kpi-val{font-size:22px;font-weight:800;color:#1e293b;}

/* risk grid */
.risk-box{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
  padding:10px;text-align:center;}
.risk-lbl{font-size:10px;text-transform:uppercase;letter-spacing:.8px;
  color:#94a3b8;margin-bottom:3px;}
.risk-val{font-size:17px;font-weight:800;}

/* signal history */
.hist-row{display:flex;justify-content:space-between;align-items:center;
  padding:4px 0;border-bottom:1px solid #f1f5f9;font-size:12px;}
.hist-row:last-child{border:none;}
.tag-BUY{background:#d1fae5;color:#065f46;padding:2px 8px;border-radius:10px;font-weight:700;font-size:11px;}
.tag-SELL{background:#fee2e2;color:#7f1d1d;padding:2px 8px;border-radius:10px;font-weight:700;font-size:11px;}
.tag-HOLD{background:#f1f5f9;color:#475569;padding:2px 8px;border-radius:10px;font-weight:700;font-size:11px;}

/* refresh bar */
.refresh-footer{background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
  padding:10px 20px;text-align:center;margin-top:14px;
  font-size:12px;color:#64748b;}

/* dataframe */
[data-testid="stDataFrame"] td{font-size:12px !important;}
[data-testid="stDataFrame"] th{font-size:11px !important;color:#64748b !important;}
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────
MT5_LOGIN    = 52869629
MT5_PASSWORD = "N&8b1KlvPSepC8"
MT5_SERVER   = "ICMarketsSC-Demo"
SYMBOL       = "EURUSD"
REFRESH_SEC  = 10

# ─── SESSION STATE INIT ───────────────────────────────────────
if "signal_history" not in st.session_state:
    st.session_state.signal_history = []
if "last_bar_time" not in st.session_state:
    st.session_state.last_bar_time = None

# ─── LOAD MODELS (once) ───────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models...")
def load_all_models():
    b      = joblib.load(MODELS_DIR / "B_scaled_v6_clean.pkl")
    bundle = joblib.load(MODELS_DIR / "deploy_ml_bundle_v18_7.pkl")
    return b["scaler"], bundle

@st.cache_resource(show_spinner="Connecting to MT5...")
def get_mt5():
    try:
        import MetaTrader5 as mt5
        mt5.shutdown()
        ok = mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
        return ok, mt5
    except ImportError:
        return False, None

# ─── LIVE DATA (no cache — fresh every cycle) ─────────────────
def fetch_live(mt5_ok, mt5_mod):
    live = False
    df   = None
    bid = ask = None
    balance = equity = pnl = 200.0
    login    = MT5_LOGIN
    positions = []

    if mt5_ok and mt5_mod:
        try:
            rates = mt5_mod.copy_rates_from_pos(SYMBOL, mt5_mod.TIMEFRAME_H4, 0, 500)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s").dt.tz_localize(None)
                live = True
            tick = mt5_mod.symbol_info_tick(SYMBOL)
            if tick:
                bid, ask = tick.bid, tick.ask
            info = mt5_mod.account_info()
            if info:
                balance = info.balance
                equity  = info.equity
                pnl     = info.profit
                login   = info.login
            pos = mt5_mod.positions_get(symbol=SYMBOL)
            positions = list(pos) if pos else []
        except Exception:
            pass

    if df is None:
        df = pd.read_csv(DATA_DIR / "eurusd_4h.csv")
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").tail(500).reset_index(drop=True)

    if bid is None:
        last = float(df["close"].iloc[-1])
        bid  = last
        ask  = round(last + 0.00009, 5)

    return df, live, bid, ask, balance, equity, pnl, login, positions

# ─── INFERENCE (no cache — fresh every cycle) ─────────────────
def run_inference(df, scaler, bundle):
    from feature_pipeline import build_features
    from regime_engine    import detect_regime
    from signal_engine    import predict_ensemble, generate_signal, build_extended_features

    feat_df  = build_features(df)
    X_raw    = feat_df.tail(1).values
    X_scaled = scaler.transform(X_raw)
    feat_row = feat_df.iloc[-1].to_dict()

    # Technical regime detection
    regime, gs, hs, gp, hp = detect_regime(feat_row)

    # Ensemble ML prediction
    X_ext = build_extended_features(X_scaled, gp, hp)
    buy_prob, sell_prob, cat_p, lgbm_p, xgb_p = predict_ensemble(X_scaled, X_ext, bundle)

    # Signal with regime gate
    signal, confidence, reason = generate_signal(buy_prob, sell_prob, regime, bundle)

    # Extra indicators for display
    atr_pips     = float(feat_df["atr_14_pips"].iloc[-1])
    adx          = float(feat_df["adx"].iloc[-1])
    rsi          = float(feat_df["rsi"].iloc[-1])
    momentum     = float(feat_df["momentum"].iloc[-1])
    range_width  = float(feat_df["range_width"].iloc[-1])
    em20_50      = float(feat_df["ema20_vs_ema50"].iloc[-1])
    em50_200     = float(feat_df["ema50_vs_ema200"].iloc[-1])

    return dict(
        signal    = signal,
        confidence= confidence,
        reason    = reason,
        buy_prob  = buy_prob,
        sell_prob = sell_prob,
        cat_buy   = float(cat_p[1]),
        cat_sell  = float(cat_p[0]),
        lgbm_buy  = float(lgbm_p[1]),
        lgbm_sell = float(lgbm_p[0]),
        xgb_buy   = float(xgb_p[1]),
        xgb_sell  = float(xgb_p[0]),
        regime    = regime,
        gmm_state = gs,
        hmm_state = hs,
        gmm_probs = gp,
        atr_pips  = atr_pips,
        adx       = adx,
        rsi       = rsi,
        momentum  = momentum,
        range_width= range_width,
        em20_50   = em20_50,
        em50_200  = em50_200,
        feat_df   = feat_df,
    )

# ─── HTML HELPERS ─────────────────────────────────────────────
def pbar(label, val_pct, bar_cls, lbl_color="#374151"):
    w = min(100, max(0, int(val_pct)))
    return f"""
<div class="pb-wrap">
  <div class="pb-row">
    <span style="color:{lbl_color}">{label}</span>
    <span style="color:{lbl_color};font-weight:800">{w}%</span>
  </div>
  <div class="pb-bg"><div class="{bar_cls}" style="width:{w}%"></div></div>
</div>"""

def card(title, body):
    return f'<div class="card"><div class="card-title">{title}</div>{body}</div>'

def kpi(label, value, color="#1e293b"):
    return f'<div class="kpi"><div class="kpi-lbl">{label}</div><div class="kpi-val" style="color:{color}">{value}</div></div>'

def risk_box(label, val, sub, color="#1e293b", bg="#f8fafc", border="#e2e8f0"):
    return f"""
<div class="risk-box" style="background:{bg};border-color:{border}">
  <div class="risk-lbl">{label}</div>
  <div class="risk-val" style="color:{color}">{val}</div>
  <div style="font-size:10px;color:#94a3b8;margin-top:2px">{sub}</div>
</div>"""

# ─── CANDLESTICK CHART ────────────────────────────────────────
def build_chart(df, signal, sl, tp, entry):
    rows = df.tail(120)
    candles_js = []
    for _, row in rows.iterrows():
        t = int(pd.Timestamp(row["time"]).timestamp())
        candles_js.append(
            f"{{time:{t},open:{row['open']:.5f},"
            f"high:{row['high']:.5f},low:{row['low']:.5f},"
            f"close:{row['close']:.5f}}}"
        )
    data_str = ",".join(candles_js)

    t0 = int(pd.Timestamp(rows["time"].iloc[0]).timestamp())
    t1 = int(pd.Timestamp(rows["time"].iloc[-1]).timestamp())

    lines = ""
    if signal in ("BUY", "SELL") and entry > 0 and sl > 0 and tp > 0:
        sl_col = "#ef4444"
        tp_col = "#10b981"
        en_col = "#3b82f6"
        lines = f"""
        const sl_ln = c.addLineSeries({{color:'{sl_col}',lineWidth:1,lineStyle:2,title:'SL {sl:.5f}'}});
        sl_ln.setData([{{time:{t0},value:{sl:.5f}}},{{time:{t1},value:{sl:.5f}}}]);
        const tp_ln = c.addLineSeries({{color:'{tp_col}',lineWidth:1,lineStyle:2,title:'TP {tp:.5f}'}});
        tp_ln.setData([{{time:{t0},value:{tp:.5f}}},{{time:{t1},value:{tp:.5f}}}]);
        const en_ln = c.addLineSeries({{color:'{en_col}',lineWidth:1,lineStyle:0,title:'Entry {entry:.5f}'}});
        en_ln.setData([{{time:{t0},value:{entry:.5f}}},{{time:{t1},value:{entry:.5f}}}]);
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/lightweight-charts/4.1.1/lightweight-charts.standalone.production.js"></script>
<style>
  html,body{{margin:0;padding:0;background:#ffffff;overflow:hidden;}}
  #chart{{width:100%;height:390px;}}
</style>
</head>
<body>
<div id="chart"></div>
<script>
(function(){{
  const el = document.getElementById('chart');
  const c = LightweightCharts.createChart(el,{{
    width:  el.offsetWidth  || 900,
    height: 390,
    layout:{{background:{{color:'#ffffff'}},textColor:'#374151'}},
    grid:{{vertLines:{{color:'#f1f5f9'}},horzLines:{{color:'#f1f5f9'}}}},
    crosshair:{{mode:LightweightCharts.CrosshairMode.Normal}},
    rightPriceScale:{{borderColor:'#e2e8f0'}},
    timeScale:{{borderColor:'#e2e8f0',timeVisible:true,secondsVisible:false}},
  }});
  const cs = c.addCandlestickSeries({{
    upColor:'#10b981',downColor:'#ef4444',
    borderUpColor:'#10b981',borderDownColor:'#ef4444',
    wickUpColor:'#10b981',wickDownColor:'#ef4444',
  }});
  cs.setData([{data_str}]);
  {lines}
  c.timeScale().scrollToRealTime();
  new ResizeObserver(()=>{{
    c.applyOptions({{width:el.offsetWidth}});
  }}).observe(el);
}})();
</script>
</body>
</html>"""
    return html

# ─── MAIN ─────────────────────────────────────────────────────
def main():
    # Load models
    try:
        scaler, bundle = load_all_models()
    except Exception as e:
        st.error(f"Model load error: {e}")
        st.stop()

    mt5_ok, mt5_mod = get_mt5()

    # Fetch live data
    with st.spinner(""):
        try:
            df, live, bid, ask, balance, equity, pnl, login, positions = \
                fetch_live(mt5_ok, mt5_mod)
        except Exception as e:
            st.error(f"Data error: {e}")
            st.stop()

    # Run inference
    with st.spinner(""):
        try:
            r = run_inference(df, scaler, bundle)
        except Exception as e:
            st.error(f"Inference error: {e}")
            st.code(traceback.format_exc())
            st.stop()

    from risk_manager import compute_sl_tp, compute_lot_size
    signal   = r["signal"]
    entry    = ask if signal == "BUY" else (bid if signal == "SELL" else (bid + ask) / 2)
    sl, tp, sl_pips = compute_sl_tp(signal, entry, r["atr_pips"])
    tp_pips  = round(abs(tp - entry) * 10000, 1)
    lot      = compute_lot_size(balance, risk_pct=1.0, sl_pips=sl_pips)
    rr       = round(tp_pips / sl_pips, 1) if sl_pips > 0 else 0
    spread_p = round((ask - bid) * 10000, 1)
    now_utc  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    regime   = r["regime"]
    gp       = r["gmm_probs"]

    # Update signal history
    cur_bar = str(df["time"].iloc[-1])
    if cur_bar != st.session_state.last_bar_time:
        st.session_state.last_bar_time = cur_bar
        st.session_state.signal_history.append({
            "time"  : datetime.now(timezone.utc).strftime("%H:%M"),
            "signal": signal,
            "buy"   : round(r["buy_prob"] * 100, 1),
            "sell"  : round(r["sell_prob"] * 100, 1),
            "regime": regime,
        })
        st.session_state.signal_history = st.session_state.signal_history[-15:]

    # ── TOP HEADER ────────────────────────────────────────────
    dot_html = '<span class="dot-live"></span>' if live else '<span class="dot-sim"></span>'
    pill_cls = "live-pill" if live else "sim-pill"
    mode_txt = "LIVE — IC Markets Demo" if live else "SIMULATION — CSV Data"
    sig_color= {"BUY":"#065f46","SELL":"#7f1d1d"}.get(signal,"#475569")
    reg_color= {"BULL_TREND":"#065f46","BEAR_TREND":"#7f1d1d",
                "RANGE_CHOP":"#78350f","ANOMALY":"#4c1d95"}.get(regime,"#475569")

    # Calculate next H4 bar time (tz-safe: strip any tz info)
    last_bar = pd.Timestamp(df["time"].iloc[-1])
    if last_bar.tzinfo is not None:
        last_bar = last_bar.tz_localize(None)
    next_bar     = last_bar + pd.Timedelta(hours=4)
    now_naive    = pd.Timestamp(datetime.utcnow())
    time_to_next = next_bar - now_naive
    mins_to_next = max(0, int(time_to_next.total_seconds() / 60))

    st.markdown(f"""
<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;
  padding:14px 22px;margin-bottom:16px;
  display:flex;align-items:center;justify-content:space-between;
  box-shadow:0 1px 4px rgba(0,0,0,.06)">
  <div>
    <div style="font-size:18px;font-weight:800;color:#0f172a">
      📈 EUR/USD H4 — AI Trading Dashboard v18.14
    </div>
    <div style="font-size:12px;color:#64748b;margin-top:3px">
      IC Markets Demo &nbsp;|&nbsp; Account #{login} &nbsp;|&nbsp; {now_utc}
      &nbsp;|&nbsp; Next H4 bar in <strong style="color:#3b82f6">{mins_to_next} min</strong>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:10px">
    <div class="{pill_cls}">{dot_html} {mode_txt}</div>
    <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:20px;
      padding:5px 14px;font-size:12px;color:#1d4ed8;font-weight:600">
      ⟳ Refresh: {REFRESH_SEC}s
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── 3-COLUMN LAYOUT ───────────────────────────────────────
    left, mid, right = st.columns([1.05, 2.1, 1.05])

    # ════════════ LEFT ════════════════════════════════════════
    with left:

        # SIGNAL BADGE
        st.markdown(f"""
<div class="card" style="text-align:center;padding-bottom:14px">
  <div class="card-title">AI Signal</div>
  <div class="badge-{signal}">{signal}</div>
  <div style="margin-top:10px;font-size:12px;font-weight:600;color:{sig_color};
    background:{'#d1fae5' if signal=='BUY' else '#fee2e2' if signal=='SELL' else '#f1f5f9'};
    border-radius:8px;padding:6px 10px;">
    {r['reason']}
  </div>
  <div style="margin-top:8px;font-size:12px;color:#64748b">
    Confidence: <strong style="color:{sig_color}">{round(r['confidence']*100,1)}%</strong>
  </div>
</div>
""", unsafe_allow_html=True)

        # PROBABILITIES
        bp_pct  = round(r["buy_prob"]  * 100, 1)
        sp_pct  = round(r["sell_prob"] * 100, 1)
        hp_pct  = round(max(0, 100 - bp_pct - sp_pct), 1)
        st.markdown(card("Signal Probabilities",
            pbar(f"BUY",  bp_pct, "pb-g", "#065f46") +
            pbar(f"SELL", sp_pct, "pb-r", "#7f1d1d") +
            pbar(f"HOLD", hp_pct, "pb-b", "#1d4ed8")
        ), unsafe_allow_html=True)

        # MODEL BREAKDOWN
        st.markdown(card("Model Breakdown (BUY %)",
            pbar(f"CatBoost  {r['cat_buy']*100:.1f}%",
                 r["cat_buy"]*100, "pb-g") +
            pbar(f"LightGBM  {r['lgbm_buy']*100:.1f}%",
                 r["lgbm_buy"]*100, "pb-o") +
            pbar(f"XGBoost   {r['xgb_buy']*100:.1f}%",
                 r["xgb_buy"]*100, "pb-p") +
            f'<div style="font-size:10px;color:#94a3b8;margin-top:6px">Signal changes every H4 bar (~4h). '
            f'Next bar in <strong style="color:#3b82f6">{mins_to_next}min</strong></div>'
        ), unsafe_allow_html=True)

        # LIVE PRICE
        st.markdown(card("Live Price — EURUSD", f"""
<div style="display:flex;justify-content:space-between;gap:8px;margin-bottom:10px">
  <div style="flex:1;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
    padding:10px;text-align:center">
    <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.8px">Bid</div>
    <div style="font-size:22px;font-weight:800;color:#065f46">{bid:.5f}</div>
  </div>
  <div style="flex:1;background:#fef2f2;border:1px solid #fecaca;border-radius:10px;
    padding:10px;text-align:center">
    <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.8px">Ask</div>
    <div style="font-size:22px;font-weight:800;color:#7f1d1d">{ask:.5f}</div>
  </div>
</div>
<div style="display:flex;justify-content:space-between;font-size:12px;color:#64748b">
  <span>Spread: <strong style="color:#1e293b">{spread_p} pips</strong></span>
  <span>Last bar: <strong style="color:#1e293b">{str(df['time'].iloc[-1])[11:16]} UTC</strong></span>
</div>
"""), unsafe_allow_html=True)

        # SIGNAL HISTORY
        hist = st.session_state.signal_history
        if hist:
            rows_html = ""
            for h in reversed(hist[-8:]):
                tag = f'<span class="tag-{h["signal"]}">{h["signal"]}</span>'
                rows_html += f"""
<div class="hist-row">
  <span style="color:#64748b">{h['time']}</span>
  {tag}
  <span style="color:#1e293b;font-size:11px">B:{h['buy']}% S:{h['sell']}%</span>
  <span style="color:#94a3b8;font-size:10px">{h['regime'][:5]}</span>
</div>"""
            st.markdown(card("Signal History (last bars)", rows_html), unsafe_allow_html=True)

    # ════════════ MIDDLE / CHART ══════════════════════════════
    with mid:

        # Chart header
        regime_pill_cls = {
            "BULL_TREND":"pill-BULL","BEAR_TREND":"pill-BEAR",
            "RANGE_CHOP":"pill-RANGE","ANOMALY":"pill-ANOM"
        }.get(regime,"pill-ANOM")
        regime_icon = {"BULL_TREND":"📈","BEAR_TREND":"📉",
                       "RANGE_CHOP":"↔","ANOMALY":"⚠"}.get(regime,"")

        adx_color = "#10b981" if r["adx"] > 25 else "#f59e0b" if r["adx"] > 20 else "#ef4444"
        rsi_color = "#ef4444" if r["rsi"] > 70 else "#10b981" if r["rsi"] < 30 else "#64748b"

        st.markdown(f"""
<div class="card" style="padding-bottom:6px">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:10px">
    <div>
      <div style="font-size:14px;font-weight:700;color:#0f172a">
        EUR/USD H4 Candlestick Chart
      </div>
      <div style="font-size:11px;color:#94a3b8;margin-top:3px">
        {len(df)} bars &nbsp;|&nbsp; Last: {str(df['time'].iloc[-1])[:16]}
        &nbsp;|&nbsp;
        <span style="color:#10b981;font-weight:600">▲ Bull candle</span>
        &nbsp;
        <span style="color:#ef4444;font-weight:600">▼ Bear candle</span>
        &nbsp;|&nbsp;
        <span style="color:#3b82f6">— Entry</span>
        &nbsp;
        <span style="color:#ef4444">-- SL</span>
        &nbsp;
        <span style="color:#10b981">-- TP</span>
      </div>
    </div>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <span style="font-size:11px;background:#f1f5f9;border-radius:8px;
        padding:3px 10px;color:{adx_color};font-weight:600">
        ADX: {r['adx']:.0f}
      </span>
      <span style="font-size:11px;background:#f1f5f9;border-radius:8px;
        padding:3px 10px;color:{rsi_color};font-weight:600">
        RSI: {r['rsi']:.0f}
      </span>
      <span style="font-size:11px;background:#f1f5f9;border-radius:8px;
        padding:3px 10px;color:#64748b;font-weight:600">
        MOM: {r['momentum']:+.2f}%
      </span>
    </div>
  </div>
""", unsafe_allow_html=True)

        # Render chart
        chart_html = build_chart(df, signal, sl, tp, entry)
        components.html(chart_html, height=400)
        st.markdown("</div>", unsafe_allow_html=True)

        # KPI row
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.markdown(kpi("Signal",     signal,               sig_color), unsafe_allow_html=True)
        k2.markdown(kpi("Confidence", f"{round(r['confidence']*100,1)}%", "#3b82f6"), unsafe_allow_html=True)
        k3.markdown(kpi("Regime",     regime[:5],           reg_color), unsafe_allow_html=True)
        k4.markdown(kpi("ATR",        f"{r['atr_pips']:.1f}p", "#1e293b"), unsafe_allow_html=True)
        k5.markdown(kpi("Spread",     f"{spread_p}p",       "#1e293b"), unsafe_allow_html=True)

        # Recent bars table — FIXED
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
                    'letter-spacing:1px;color:#94a3b8;margin-bottom:6px">'
                    'Recent H4 Bars (candle direction = close vs open)</div>',
                    unsafe_allow_html=True)

        recent = df.tail(10)[["time","open","high","low","close","tick_volume"]].copy()
        # Change = close vs PREVIOUS close (price move between bars)
        recent["prev_close"] = recent["close"].shift(1)
        recent["Δ pips"] = ((recent["close"] - recent["prev_close"]) * 10000).round(1)
        # Candle dir = close vs open (within the bar)
        recent["Candle"] = (recent["close"] - recent["open"]).apply(
            lambda x: "▲ Bull" if x >= 0 else "▼ Bear"
        )
        # Bar move = Δ pips direction
        recent["Bar Δ"] = recent["Δ pips"].apply(
            lambda x: f"+{x:.1f}" if x >= 0 else f"{x:.1f}"
        )
        recent["Time"] = recent["time"].astype(str).str[:16]
        recent = recent.rename(columns={
            "open":"Open","high":"High","low":"Low","close":"Close","tick_volume":"Volume"
        })
        display = recent[["Time","Open","High","Low","Close","Volume","Bar Δ","Candle"]].copy()
        display = display.iloc[::-1].reset_index(drop=True)

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Open":   st.column_config.NumberColumn(format="%.5f"),
                "High":   st.column_config.NumberColumn(format="%.5f"),
                "Low":    st.column_config.NumberColumn(format="%.5f"),
                "Close":  st.column_config.NumberColumn(format="%.5f"),
                "Bar Δ":  st.column_config.TextColumn(),
                "Candle": st.column_config.TextColumn(),
            }
        )
        st.markdown(
            '<div style="font-size:10px;color:#94a3b8;margin-top:4px">'
            '⚠ Candle dir = within-bar direction (close vs open). '
            'Trend regime = multi-bar EMA alignment. Both can differ simultaneously.</div>',
            unsafe_allow_html=True
        )

    # ════════════ RIGHT ═══════════════════════════════════════
    with right:

        # REGIME
        pill_map = {"BULL_TREND":"pill-BULL","BEAR_TREND":"pill-BEAR",
                    "RANGE_CHOP":"pill-RANGE","ANOMALY":"pill-ANOM"}
        pill_cls2 = pill_map.get(regime,"pill-ANOM")
        icon2 = {"BULL_TREND":"📈","BEAR_TREND":"📉","RANGE_CHOP":"↔️","ANOMALY":"⚠️"}.get(regime,"")

        em_bull  = r["em20_50"] > 0 and r["em50_200"] > 0
        em_bear  = r["em20_50"] < 0 and r["em50_200"] < 0
        em20_col = "#10b981" if r["em20_50"] > 0 else "#ef4444"
        em50_col = "#10b981" if r["em50_200"] > 0 else "#ef4444"

        st.markdown(card("Market Regime", f"""
<div class="pill {pill_cls2}" style="margin-bottom:12px;font-size:17px">
  {icon2} {regime}
</div>
{pbar("BULL",  round(gp[0]*100,1), "pb-g")}
{pbar("BEAR",  round(gp[1]*100,1), "pb-r")}
{pbar("RANGE", round(gp[2]*100,1), "pb-o")}
{pbar("ANOM",  round(gp[3]*100,1), "pb-gray")}
<div style="margin-top:10px;font-size:11px;color:#64748b;
  background:#f8fafc;border-radius:8px;padding:8px">
  <div style="display:flex;justify-content:space-between;margin-bottom:3px">
    <span>ADX</span>
    <strong style="color:{'#10b981' if r['adx']>25 else '#f59e0b'}">{r['adx']:.1f}</strong>
  </div>
  <div style="display:flex;justify-content:space-between;margin-bottom:3px">
    <span>EMA20 vs 50</span>
    <strong style="color:{em20_col}">{r['em20_50']:+.1f}p</strong>
  </div>
  <div style="display:flex;justify-content:space-between;margin-bottom:3px">
    <span>EMA50 vs 200</span>
    <strong style="color:{em50_col}">{r['em50_200']:+.1f}p</strong>
  </div>
  <div style="display:flex;justify-content:space-between">
    <span>RSI</span>
    <strong style="color:{'#ef4444' if r['rsi']>70 else '#10b981' if r['rsi']<30 else '#64748b'}">{r['rsi']:.1f}</strong>
  </div>
</div>
"""), unsafe_allow_html=True)

        # RISK MANAGEMENT
        sl_color = "#ef4444"
        tp_color = "#10b981"
        rr_color = "#3b82f6" if rr >= 2 else "#f59e0b" if rr >= 1.5 else "#ef4444"

        st.markdown(card("Risk Management", f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px">
  {risk_box("ATR", f"{r['atr_pips']:.1f}p", "Average True Range")}
  {risk_box("Lot Size", str(lot), "1% risk")}
  {risk_box("Stop Loss", f"{sl:.5f}",
    f"{sl_pips:.1f} pips below" if signal=='BUY' else f"{sl_pips:.1f} pips above",
    sl_color, "#fff5f5", "#fecaca")}
  {risk_box("Take Profit", f"{tp:.5f}",
    f"{tp_pips:.1f} pips above" if signal=='BUY' else f"{tp_pips:.1f} pips below",
    tp_color, "#f0fdf4", "#bbf7d0")}
</div>
<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
  padding:12px;text-align:center">
  <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;
    letter-spacing:.8px;margin-bottom:4px">Risk / Reward Ratio</div>
  <div style="font-size:26px;font-weight:800;color:{rr_color}">1 : {rr}</div>
</div>
"""), unsafe_allow_html=True)

        # ACCOUNT
        pnl_color = "#10b981" if pnl >= 0 else "#ef4444"
        pnl_sign  = "+" if pnl >= 0 else ""
        st.markdown(card(f"Account #{login}", f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
  {risk_box("Balance",   f"${balance:,.2f}", "Account balance")}
  {risk_box("Equity",    f"${equity:,.2f}",  "Current equity")}
  {risk_box("Open P&L",  f"{pnl_sign}${abs(pnl):.2f}",
    "Floating P&L", pnl_color,
    "#f0fdf4" if pnl>=0 else "#fff5f5",
    "#bbf7d0" if pnl>=0 else "#fecaca")}
  {risk_box("Positions", str(len(positions)), "Open trades")}
</div>
"""), unsafe_allow_html=True)

        # SYSTEM INFO
        gate = bundle["regime_prob_gates"].get(
            {"BULL_TREND":0,"BEAR_TREND":1,"RANGE_CHOP":2,"ANOMALY":3}.get(regime,0),
            bundle["prob_gate"])
        st.markdown(card("System Info", f"""
<div style="font-size:12px;color:#374151">
  <div style="display:flex;justify-content:space-between;padding:4px 0;
    border-bottom:1px solid #f1f5f9">
    <span style="color:#64748b">Engine</span>
    <strong>Cat + LGBM + XGB</strong>
  </div>
  <div style="display:flex;justify-content:space-between;padding:4px 0;
    border-bottom:1px solid #f1f5f9">
    <span style="color:#64748b">Regime Engine</span>
    <strong>ADX + EMA + ATR</strong>
  </div>
  <div style="display:flex;justify-content:space-between;padding:4px 0;
    border-bottom:1px solid #f1f5f9">
    <span style="color:#64748b">Features</span>
    <strong>103 + 10</strong>
  </div>
  <div style="display:flex;justify-content:space-between;padding:4px 0;
    border-bottom:1px solid #f1f5f9">
    <span style="color:#64748b">Prob Gate</span>
    <strong style="color:#3b82f6">{gate:.2f} ({regime[:5]})</strong>
  </div>
  <div style="display:flex;justify-content:space-between;padding:4px 0;
    border-bottom:1px solid #f1f5f9">
    <span style="color:#64748b">H4 Bar Change</span>
    <strong style="color:#3b82f6">{mins_to_next} min</strong>
  </div>
  <div style="display:flex;justify-content:space-between;padding:4px 0">
    <span style="color:#64748b">Last Update</span>
    <strong style="color:#10b981">{now_utc[11:19]}</strong>
  </div>
</div>
"""), unsafe_allow_html=True)

    # ── AUTO-REFRESH FOOTER ───────────────────────────────────
    st.markdown(f"""
<div class="refresh-footer">
  Dashboard auto-refreshes every <strong style="color:#3b82f6">{REFRESH_SEC} seconds</strong>
  &nbsp;|&nbsp; Signal updates every <strong style="color:#3b82f6">H4 bar (~4 hours)</strong>
  &nbsp;|&nbsp; Next bar in <strong style="color:#3b82f6">{mins_to_next} min</strong>
  &nbsp;|&nbsp;
  <span style="color:#94a3b8">⚠ Bull candles ≠ Bull trend. Candle = single bar. Trend = EMA alignment.</span>
</div>
<script>
(function(){{
  let t={REFRESH_SEC};
  const update=()=>{{
    const els=document.querySelectorAll('[data-countdown]');
    els.forEach(e=>e.textContent=t+'s');
  }};
  const iv=setInterval(()=>{{t--;update();if(t<=0)clearInterval(iv);}},1000);
}})();
</script>
""", unsafe_allow_html=True)

    time.sleep(REFRESH_SEC)
    st.rerun()


if __name__ == "__main__":
    main()
