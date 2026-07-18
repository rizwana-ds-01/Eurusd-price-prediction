"""
live_dashboard.py  —  EUR/USD H4  v18.14
=========================================
CORRECT CMD:
    python live_dashboard.py

Dashboard → http://localhost:5000
"""

import os, sys, time, json, logging, pickle, threading, traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string

# ── path setup ───────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from feature_pipeline import build_features
from regime_engine    import load_regime_models, detect_regime
from signal_engine    import load_signal_bundle, predict_signal, generate_signal
from risk_manager     import compute_sl_tp, compute_lot_size

# ── directories ──────────────────────────────────────────────────────────────
LOGS_DIR   = HERE / "logs"
MODELS_DIR = HERE / "models"
DATA_DIR   = HERE / "data"
for d in (LOGS_DIR, MODELS_DIR, DATA_DIR):
    d.mkdir(exist_ok=True)

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "dashboard.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── IC Markets MT5 config ─────────────────────────────────────────────────────
MT5_LOGIN    = 52869629
MT5_PASSWORD = "N&8b1KlvPSepC8"
MT5_SERVER   = "ICMarketsSC-Demo"
SYMBOL       = "EURUSD"

# ── model paths ───────────────────────────────────────────────────────────────
BUNDLE_B      = MODELS_DIR / "B_scaled_v6_clean.pkl"
BUNDLE_D      = MODELS_DIR / "D_clusters_v9.pkl"
BUNDLE_DEPLOY = MODELS_DIR / "deploy_ml_bundle_v18_7.pkl"

# ── shared state ──────────────────────────────────────────────────────────────
state = {
    "connected": False, "last_update": "—", "symbol": SYMBOL,
    "bid": 0.0, "ask": 0.0, "spread_pips": 0.0,
    "signal": "—", "signal_color": "#888", "confidence": 0.0,
    "reason": "—", "up_prob": 0.0, "down_prob": 0.0,
    "regime": "—", "regime_color": "#888",
    "gmm_state": -1, "hmm_state": -1,
    "gmm_probs": [0,0,0,0], "hmm_probs": [0,0,0,0],
    "atr_pips": 0.0, "sl_price": 0.0, "tp_price": 0.0, "sl_pips": 0.0, "lot": 0.10,
    "account_balance": 10000.0, "account_equity": 10000.0,
    "account_profit": 0.0, "open_positions": 0,
    "candles": [], "errors": [], "inference_ms": 0,
}


def load_models():
    import joblib
    log.info("Loading models …")
    b      = joblib.load(BUNDLE_B)
    scaler = b["scaler"]
    gmm, hmm, dim_scaler, dim_pca, good_dims, regime_names = load_regime_models(BUNDLE_D)
    bun    = load_signal_bundle(BUNDLE_DEPLOY)
    log.info("Models loaded ✓  (CatBoost+LightGBM+XGBoost ensemble)")
    return scaler, gmm, hmm, dim_scaler, dim_pca, good_dims, regime_names, bun


def mt5_connect():
    try:
        import MetaTrader5 as mt5
        mt5.shutdown()
        ok = mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
        if ok:
            log.info("MT5 connected — %s", mt5.account_info())
            return True, mt5
        log.error("MT5 init failed: %s", mt5.last_error())
        return False, None
    except ImportError:
        log.warning("MetaTrader5 not installed — SIMULATION MODE (CSV)")
        return False, None


def fetch_bars_mt5(mt5mod, n=500):
    rates = mt5mod.copy_rates_from_pos(SYMBOL, mt5mod.TIMEFRAME_H4, 0, n)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def load_csv_data():
    for p in [DATA_DIR/"eurusd_4h.csv", HERE/"eurusd_4h.csv", Path("eurusd_4h.csv")]:
        if p.exists():
            df = pd.read_csv(p)
            df["time"] = pd.to_datetime(df["time"])
            if "tick_volume" not in df.columns and "real_volume" in df.columns:
                df["tick_volume"] = df["real_volume"]
            if "spread" not in df.columns:
                df["spread"] = 1.0
            return df.tail(500).reset_index(drop=True)
    raise FileNotFoundError("eurusd_4h.csv not found — place in data/ folder")


def inference_loop(scaler, gmm, hmm, dim_scaler, dim_pca,
                   good_dims, regime_names, bun, mt5_ok, mt5_module):
    global state
    REGIME_COLORS = {
        "BULL_TREND": "#00e676", "BEAR_TREND": "#ff5252",
        "RANGE_CHOP": "#ffd740", "ANOMALY": "#ff6d00", "UNKNOWN": "#888",
    }
    while True:
        t0 = time.time()
        try:
            # ── fetch data ────────────────────────────────────────────────
            if mt5_ok and mt5_module is not None:
                df = fetch_bars_mt5(mt5_module)
                if df is None:
                    raise RuntimeError("No bars from MT5")
                tick  = mt5_module.symbol_info_tick(SYMBOL)
                bid   = tick.bid if tick else 0.0
                ask   = tick.ask if tick else 0.0
                info  = mt5_module.account_info()
                bal   = info.balance if info else 10000.0
                eq    = info.equity  if info else 10000.0
                prof  = info.profit  if info else 0.0
                pos   = mt5_module.positions_get(symbol=SYMBOL)
                n_pos = len(pos) if pos else 0
                connected = True
            else:
                df = load_csv_data()
                last = float(df["close"].iloc[-1])
                bid, ask = last, last + 0.0002
                bal, eq, prof, n_pos = 10000.0, 10000.0, 0.0, 0
                connected = False

            # ── feature engineering ───────────────────────────────────────
            feat_df  = build_features(df)
            X_raw    = feat_df.tail(1).values        # (1, 103)
            X_scaled = scaler.transform(X_raw)       # (1, 103)
            feat_cols = list(feat_df.columns)

            # ── regime detection ──────────────────────────────────────────
            regime_str, gmm_state, hmm_state, gmm_probs, hmm_probs = \
                detect_regime(X_scaled, gmm, hmm, dim_scaler, dim_pca,
                              good_dims, regime_names)

            # ── ensemble signal (CatBoost + LightGBM + XGBoost) ──────────
            up_prob, down_prob = predict_signal(X_scaled, bun, feat_cols)
            signal, confidence, reason = generate_signal(
                up_prob, down_prob, regime_str, gmm_state, bun)

            # ── risk ──────────────────────────────────────────────────────
            atr_pips = float(feat_df["atr_14_pips"].iloc[-1])
            entry    = ask if signal == "BUY" else bid
            sl, tp, sl_pips = compute_sl_tp(signal, entry, atr_pips)
            lot = compute_lot_size(bal, risk_pct=1.0, sl_pips=max(sl_pips, 1.0))

            # ── candles for chart ─────────────────────────────────────────
            candles = [
                {"t": str(r["time"]),
                 "o": round(float(r["open"]),  5),
                 "h": round(float(r["high"]),  5),
                 "l": round(float(r["low"]),   5),
                 "c": round(float(r["close"]), 5)}
                for _, r in df.tail(80).iterrows()
            ]

            sig_color   = {"BUY": "#00e676", "SELL": "#ff5252"}.get(signal, "#ffd740")
            spread_pips = round((ask - bid) * 10000, 1)
            ms          = int((time.time() - t0) * 1000)

            state.update({
                "connected":       connected,
                "last_update":     datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "bid":             round(bid, 5),
                "ask":             round(ask, 5),
                "spread_pips":     spread_pips,
                "signal":          signal,
                "signal_color":    sig_color,
                "confidence":      round(confidence * 100, 1),
                "reason":          reason,
                "up_prob":         round(up_prob   * 100, 1),
                "down_prob":       round(down_prob * 100, 1),
                "regime":          regime_str,
                "regime_color":    REGIME_COLORS.get(regime_str, "#888"),
                "gmm_state":       gmm_state,
                "hmm_state":       hmm_state,
                "gmm_probs":       [round(p * 100, 1) for p in gmm_probs],
                "hmm_probs":       [round(p * 100, 1) for p in hmm_probs],
                "atr_pips":        round(atr_pips, 1),
                "sl_price":        sl,
                "tp_price":        tp,
                "sl_pips":         round(sl_pips, 1),
                "lot":             lot,
                "account_balance": round(bal,  2),
                "account_equity":  round(eq,   2),
                "account_profit":  round(prof, 2),
                "open_positions":  n_pos,
                "candles":         candles,
                "inference_ms":    ms,
                "errors":          [],
            })

            log.info("[%s] %s | UP=%.1f%% DOWN=%.1f%% | %s | %dms",
                     state["last_update"], signal,
                     up_prob*100, down_prob*100, regime_str, ms)

        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            log.error("loop error: %s\n%s", err, traceback.format_exc())
            state["errors"] = (state["errors"] + [err])[-5:]

        time.sleep(10)


# ── Flask ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EUR/USD H4 — AI Dashboard v18.14</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  :root{
    --bg:#060b14;--panel:#0d1526;--border:#1e2d45;
    --green:#00e676;--red:#ff5252;--yellow:#ffd740;--orange:#ff6d00;
    --accent:#4da6ff;--muted:#64748b;--text:#e2e8f0;
  }
  html,body{height:100%;background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;font-size:13px}

  #topbar{display:flex;align-items:center;gap:12px;padding:9px 18px;
    background:var(--panel);border-bottom:1px solid var(--border);position:sticky;top:0;z-index:10}
  #topbar h1{font-size:14px;font-weight:700;color:var(--accent);letter-spacing:.4px}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--muted);flex-shrink:0}
  .dot.live{background:var(--green);box-shadow:0 0 6px var(--green);animation:pulse 1.4s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
  #status-txt{font-size:11px;color:var(--muted)}
  #clock{margin-left:auto;font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums}
  #spin{width:14px;height:14px;border:2px solid var(--border);border-top-color:var(--accent);
    border-radius:50%}
  #topbar.refreshing #spin{animation:rot .6s linear infinite}
  @keyframes rot{to{transform:rotate(360deg)}}
  #err-box{display:none;background:#2d0a0a;border:1px solid var(--red);color:var(--red);
    padding:5px 14px;font-size:11px;margin:5px 18px;border-radius:4px}

  #layout{display:grid;grid-template-columns:268px 1fr;height:calc(100vh - 46px);overflow:hidden}

  #sidebar{background:var(--panel);border-right:1px solid var(--border);
    overflow-y:auto;padding:10px 8px;display:flex;flex-direction:column;gap:8px}
  .card{background:#0a1221;border:1px solid var(--border);border-radius:7px;padding:10px 12px}
  .ct{font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:1px;
    color:var(--accent);margin-bottom:8px}
  .sr{display:flex;justify-content:space-between;align-items:center;
    padding:3px 0;border-bottom:1px solid #131f33}
  .sr:last-child{border-bottom:none}
  .sl{color:var(--muted);font-size:11px}
  .sv{font-weight:600;font-size:12px;font-variant-numeric:tabular-nums}

  #sig-badge{font-size:26px;font-weight:900;letter-spacing:2px;padding:5px 16px;
    border-radius:6px;border:2px solid #888;color:#888;display:inline-block}
  #sig-reason{font-size:10px;color:var(--muted);margin-top:3px;text-align:center}

  .pb{display:flex;align-items:center;gap:5px;margin:2px 0}
  .pl{font-size:10px;color:var(--muted);width:28px}
  .pt{flex:1;background:#111e33;border-radius:3px;height:5px;overflow:hidden}
  .pf{height:100%;border-radius:3px;transition:width .5s}
  #up-bar{background:var(--green)} #dn-bar{background:var(--red)}
  .pp{font-size:10px;font-variant-numeric:tabular-nums;width:34px;text-align:right}

  .rg{display:grid;grid-template-columns:1fr 1fr;gap:3px;margin-top:5px}
  .rc{background:#060b14;border:1px solid var(--border);border-radius:4px;padding:4px 7px}
  .rn{font-size:9px;color:var(--muted)} .rv{font-size:12px;font-weight:700;color:var(--accent)}

  .ag{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-top:4px}
  .ac{background:#060b14;border:1px solid var(--border);border-radius:4px;padding:5px 7px}
  .ac .lb{font-size:9px;color:var(--muted)} .ac .vl{font-size:13px;font-weight:700}

  #main{display:flex;flex-direction:column;overflow:hidden}
  .ch{display:flex;align-items:center;justify-content:space-between;
    padding:8px 14px;border-bottom:1px solid var(--border);background:var(--panel)}
  .ch h3{font-size:13px;font-weight:700}
  #cw{flex:1;overflow:hidden;padding:6px}
  #chart{width:100%;height:100%;min-height:280px}

  #bb{display:flex;border-top:1px solid var(--border);background:var(--panel)}
  .bs{flex:1;padding:8px 14px;border-right:1px solid var(--border);text-align:center}
  .bs:last-child{border-right:none}
  .bs .lb{font-size:9.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px}
  .bs .vl{font-size:14px;font-weight:700;margin-top:1px}
</style>
</head>
<body>

<div id="topbar">
  <div id="spin"></div>
  <h1>EUR/USD H4 &mdash; AI Dashboard v18.14</h1>
  <div class="dot" id="status-dot"></div>
  <span id="status-txt">Connecting…</span>
  <span id="clock"></span>
</div>
<div id="err-box"></div>

<div id="layout">
  <div id="sidebar">

    <div class="card" style="text-align:center">
      <div class="ct">Signal</div>
      <div id="sig-badge">—</div>
      <div id="sig-reason"></div>
      <div style="margin-top:8px">
        <div class="pb"><span class="pl" style="color:var(--green)">UP</span>
          <div class="pt"><div class="pf" id="up-bar" style="width:0%"></div></div>
          <span class="pp" id="up-pct">0%</span></div>
        <div class="pb"><span class="pl" style="color:var(--red)">DN</span>
          <div class="pt"><div class="pf" id="dn-bar" style="width:0%"></div></div>
          <span class="pp" id="dn-pct">0%</span></div>
      </div>
    </div>

    <div class="card">
      <div class="ct">Live Price</div>
      <div class="sr"><span class="sl">BID</span><span class="sv" id="bid">—</span></div>
      <div class="sr"><span class="sl">ASK</span><span class="sv" id="ask">—</span></div>
      <div class="sr"><span class="sl">Spread</span><span class="sv" id="spread">—</span></div>
    </div>

    <div class="card">
      <div class="ct">Market Regime</div>
      <div style="font-size:17px;font-weight:800;margin-bottom:5px" id="regime-name">—</div>
      <div class="sr"><span class="sl">GMM State</span><span class="sv" id="gmm-state">—</span></div>
      <div class="sr"><span class="sl">HMM State</span><span class="sv" id="hmm-state">—</span></div>
      <div class="rg">
        <div class="rc"><div class="rn">BULL</div><div class="rv" id="gp0">0%</div></div>
        <div class="rc"><div class="rn">BEAR</div><div class="rv" id="gp1">0%</div></div>
        <div class="rc"><div class="rn">RANGE</div><div class="rv" id="gp2">0%</div></div>
        <div class="rc"><div class="rn">ANOMALY</div><div class="rv" id="gp3">0%</div></div>
      </div>
    </div>

    <div class="card">
      <div class="ct">Risk Parameters</div>
      <div class="sr"><span class="sl">ATR (pips)</span><span class="sv" id="atr">—</span></div>
      <div class="sr"><span class="sl">Stop Loss</span><span class="sv" style="color:var(--red)" id="sl">—</span></div>
      <div class="sr"><span class="sl">Take Profit</span><span class="sv" style="color:var(--green)" id="tp">—</span></div>
      <div class="sr"><span class="sl">SL (pips)</span><span class="sv" id="sl-pips">—</span></div>
      <div class="sr"><span class="sl">Lot Size</span><span class="sv" id="lot">—</span></div>
    </div>

    <div class="card">
      <div class="ct">Account — IC Markets Demo</div>
      <div class="ag">
        <div class="ac"><div class="lb">Balance</div><div class="vl" id="balance">—</div></div>
        <div class="ac"><div class="lb">Equity</div><div class="vl" id="equity">—</div></div>
        <div class="ac"><div class="lb">P&L</div><div class="vl" id="profit">—</div></div>
        <div class="ac"><div class="lb">Positions</div><div class="vl" id="positions">—</div></div>
      </div>
    </div>

    <div class="card">
      <div class="ct">System</div>
      <div class="sr"><span class="sl">Inference</span><span class="sv" id="inf-ms">—</span></div>
      <div class="sr"><span class="sl">Models</span><span class="sv" style="color:var(--accent)">CAT+LGB+XGB</span></div>
      <div class="sr"><span class="sl">Refresh</span><span class="sv">10 s</span></div>
    </div>

  </div>

  <div id="main">
    <div class="ch">
      <h3>EUR/USD H4 &mdash; Candlestick (Live MT5)</h3>
      <span style="font-size:11px;color:var(--muted)" id="bar-count">—</span>
    </div>
    <div id="cw"><div id="chart"></div></div>
    <div id="bb">
      <div class="bs"><div class="lb">Signal</div><div class="vl" id="bb-sig" style="color:#888">—</div></div>
      <div class="bs"><div class="lb">Confidence</div><div class="vl" id="bb-conf">—</div></div>
      <div class="bs"><div class="lb">Regime</div><div class="vl" id="bb-reg" style="color:#888">—</div></div>
      <div class="bs"><div class="lb">Reason</div><div class="vl" style="font-size:10px" id="bb-rsn">—</div></div>
      <div class="bs"><div class="lb">Updated</div><div class="vl" style="font-size:11px" id="bb-upd">—</div></div>
    </div>
  </div>
</div>

<script>
const chartEl = document.getElementById("chart");
const chart = LightweightCharts.createChart(chartEl, {
  width: chartEl.offsetWidth||900, height: chartEl.offsetHeight||380,
  layout:{background:{color:"#060b14"},textColor:"#e2e8f0"},
  grid:{vertLines:{color:"#111e33"},horzLines:{color:"#111e33"}},
  crosshair:{mode:LightweightCharts.CrosshairMode.Normal},
  rightPriceScale:{borderColor:"#1e2d45"},
  timeScale:{borderColor:"#1e2d45",timeVisible:true,secondsVisible:false},
});
const cs = chart.addCandlestickSeries({
  upColor:"#00e676",downColor:"#ff5252",
  borderUpColor:"#00e676",borderDownColor:"#ff5252",
  wickUpColor:"#00e676",wickDownColor:"#ff5252",
});
new ResizeObserver(()=>chart.applyOptions({width:chartEl.offsetWidth,height:chartEl.offsetHeight})).observe(chartEl);

const g = id=>document.getElementById(id);
function tx(id,txt,col){const e=g(id);if(!e)return;e.textContent=txt;if(col)e.style.color=col;}
function sw(id,p){const e=g(id);if(e)e.style.width=Math.min(100,Math.max(0,p))+"%";}
const f5=v=>Number(v).toFixed(5);
const fusd=v=>"$"+Number(v).toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2});

function applyData(d){
  const live=d.connected;
  g("status-dot").className="dot"+(live?" live":"");
  tx("status-txt",live?"LIVE — IC Markets #52869629":"SIMULATION (CSV)");

  tx("sig-badge",d.signal,d.signal_color);
  g("sig-badge").style.borderColor=d.signal_color||"#888";
  tx("sig-reason",d.reason||"");
  tx("up-pct",d.up_prob+"%"); sw("up-bar",d.up_prob);
  tx("dn-pct",d.down_prob+"%"); sw("dn-bar",d.down_prob);

  tx("bid",f5(d.bid)); tx("ask",f5(d.ask)); tx("spread",d.spread_pips+" pips");

  tx("regime-name",d.regime,d.regime_color);
  tx("gmm-state",d.gmm_state); tx("hmm-state",d.hmm_state);
  const gp=d.gmm_probs||[0,0,0,0];
  tx("gp0",gp[0]+"%"); tx("gp1",gp[1]+"%"); tx("gp2",gp[2]+"%"); tx("gp3",gp[3]+"%");

  tx("atr",d.atr_pips+" pips");
  tx("sl",f5(d.sl_price)); tx("tp",f5(d.tp_price));
  tx("sl-pips",d.sl_pips+" pips"); tx("lot",d.lot);

  tx("balance",fusd(d.account_balance)); tx("equity",fusd(d.account_equity));
  tx("profit",fusd(d.account_profit),d.account_profit>=0?"#00e676":"#ff5252");
  tx("positions",d.open_positions);
  tx("inf-ms",d.inference_ms+" ms");

  tx("bb-sig",d.signal,d.signal_color);
  tx("bb-conf",d.confidence+"%");
  tx("bb-reg",d.regime,d.regime_color);
  tx("bb-rsn",d.reason||"—");
  tx("bb-upd",(d.last_update||"—").slice(11,19)+" UTC");

  if(d.candles&&d.candles.length){
    const bars=d.candles.map(c=>({
      time:Math.floor(new Date(c.t).getTime()/1000),
      open:c.o,high:c.h,low:c.l,close:c.c
    }));
    const seen=new Set();
    const u=bars.filter(b=>{if(seen.has(b.time))return false;seen.add(b.time);return true;});
    u.sort((a,b)=>a.time-b.time);
    cs.setData(u);
    chart.timeScale().scrollToRealTime();
    tx("bar-count",u.length+" bars");
  }
  if(d.errors&&d.errors.length){
    g("err-box").style.display="block";
    g("err-box").textContent="Error: "+d.errors[d.errors.length-1];
  } else {
    g("err-box").style.display="none";
  }
}

async function refresh(){
  g("topbar").classList.add("refreshing");
  try{const r=await fetch("/api/state");applyData(await r.json());}
  catch(e){console.error(e);}
  finally{g("topbar").classList.remove("refreshing");}
}
refresh();
setInterval(refresh,10000);
setInterval(()=>{g("clock").textContent=new Date().toUTCString().slice(0,25)+" UTC";},1000);
</script>
</body></html>
"""

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route("/api/state")
def api_state():
    return jsonify(state)


if __name__ == "__main__":
    scaler, gmm, hmm, dim_scaler, dim_pca, good_dims, regime_names, bun = load_models()
    mt5_ok, mt5_module = mt5_connect()

    t = threading.Thread(
        target=inference_loop,
        args=(scaler, gmm, hmm, dim_scaler, dim_pca,
              good_dims, regime_names, bun, mt5_ok, mt5_module),
        daemon=True,
    )
    t.start()
    log.info("Inference thread started ✓")
    log.info("Dashboard → http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
