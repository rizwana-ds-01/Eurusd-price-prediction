"""
live_dashboard.py  —  EUR/USD H4  v18.14
=========================================
Flask dashboard that:
  • Connects to IC Markets MT5 Demo
  • Pulls live H4 bars every 10 s
  • Runs feature engineering → regime detection → signal
  • Serves a beautiful dark trading dashboard at http://localhost:5000
  • Auto-refreshes data via AJAX every 10 s (no page reload)

Usage:
    python app/live_dashboard.py
"""

import os
import sys
import time
import json
import logging
import pickle
import joblib
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from flask import Flask, jsonify, render_template_string

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from feature_pipeline import build_features
from regime_engine    import load_regime_models, detect_regime
from signal_engine    import load_udl_engine, predict_udl, generate_signal
from risk_manager     import compute_sl_tp, compute_lot_size
st.write("Dashboard started successfully")
# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "logs" / "dashboard.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ── MT5 config ───────────────────────────────────────────────────────────────
MT5_LOGIN    = 52869629
MT5_PASSWORD = "N&8b1KlvPSepC8"
MT5_SERVER   = "ICMarketsSC-Demo"
SYMBOL       = "EURUSD"

# ── model paths ──────────────────────────────────────────────────────────────
MODELS = ROOT / "models"
BUNDLE_B   = MODELS / "B_scaled_v6_clean.pkl"
BUNDLE_D   = MODELS / "D_clusters_v9.pkl"
BUNDLE_UDL = MODELS / "udl_engine_v3.pkl"

# ── global state shared between bg thread and Flask ──────────────────────────
state = {
    "connected"   : False,
    "last_update" : "—",
    "symbol"      : SYMBOL,
    "bid"         : 0.0,
    "ask"         : 0.0,
    "spread_pips" : 0.0,
    "signal"      : "—",
    "signal_color": "#888",
    "confidence"  : 0.0,
    "buy_prob"    : 0.0,
    "sell_prob"   : 0.0,
    "regime"      : "—",
    "regime_color": "#888",
    "gmm_state"   : -1,
    "hmm_state"   : -1,
    "gmm_probs"   : [0, 0, 0, 0],
    "hmm_probs"   : [0, 0, 0, 0],
    "atr_pips"    : 0.0,
    "sl_price"    : 0.0,
    "tp_price"    : 0.0,
    "sl_pips"     : 0.0,
    "lot"         : 0.10,
    "account_balance": 10000.0,
    "account_equity" : 10000.0,
    "account_profit" : 0.0,
    "open_positions" : 0,
    "candles"     : [],
    "errors"      : [],
    "inference_ms": 0,
}


# ── load models ──────────────────────────────────────────────────────────────
def load_models():
    log.info("Loading models …")
    b = joblib.load(BUNDLE_B)
    scaler = b["scaler"]

    gmm, hmm, dim_scaler, dim_pca, good_dims, regime_names = load_regime_models(BUNDLE_D)
    udl = load_udl_engine(BUNDLE_UDL)

    log.info("Models loaded ✓")
    return scaler, gmm, hmm, dim_scaler, dim_pca, good_dims, regime_names, udl
st.write("Before load_models()")

scaler, gmm, hmm, dim_scaler, dim_pca, good_dims, regime_names, udl = load_models()

st.write("After load_models()")

# ── MT5 helpers ──────────────────────────────────────────────────────────────
def mt5_connect():
    try:
        import MetaTrader5 as mt5
        mt5.shutdown()
        ok = mt5.initialize(
            login   = MT5_LOGIN,
            password= MT5_PASSWORD,
            server  = MT5_SERVER,
        )
        if ok:
            log.info("MT5 connected — %s", mt5.account_info())
            return True, mt5
        else:
            log.error("MT5 init failed: %s", mt5.last_error())
            return False, mt5
    except ImportError:
        log.warning("MetaTrader5 not available — SIMULATION MODE")
        return False, None


def fetch_bars(mt5, symbol=SYMBOL, n=500):
    import MetaTrader5 as mt5mod
    rates = mt5mod.copy_rates_from_pos(symbol, mt5mod.TIMEFRAME_H4, 0, n)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def fetch_account(mt5mod):
    info = mt5mod.account_info()
    if info:
        return info.balance, info.equity, info.profit
    return state["account_balance"], state["account_equity"], state["account_profit"]


def fetch_positions(mt5mod, symbol=SYMBOL):
    pos = mt5mod.positions_get(symbol=symbol)
    return len(pos) if pos else 0


# ── simulation fallback ───────────────────────────────────────────────────────
def load_csv_data():
    csv_path = ROOT / "data" / "eurusd_4h.csv"
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])
    return df.tail(500).reset_index(drop=True)


def simulate_tick(df):
    """Return a simulated bid/ask from last close."""
    last_close = float(df["close"].iloc[-1])
    spread = 0.0002
    bid = last_close
    ask = last_close + spread
    return bid, ask


# ── background inference loop ─────────────────────────────────────────────────
def inference_loop(scaler, gmm, hmm, dim_scaler, dim_pca,
                   good_dims, regime_names, udl, mt5_ok, mt5_module):
    global state

    import MetaTrader5 as mt5mod_global

    while True:
        t0 = time.time()
        try:
            # ── fetch data ───────────────────────────────────────────────
            if mt5_ok:
                df = fetch_bars(mt5_module, SYMBOL)
                if df is None:
                    raise RuntimeError("No bars returned from MT5")
                tick = mt5mod_global.symbol_info_tick(SYMBOL)
                bid  = tick.bid if tick else 0.0
                ask  = tick.ask if tick else 0.0
                bal, eq, profit = fetch_account(mt5mod_global)
                n_pos = fetch_positions(mt5mod_global)
            else:
                df     = load_csv_data()
                bid, ask = simulate_tick(df)
                bal, eq, profit = 10000.0, 10000.0, 0.0
                n_pos  = 0

            # ── features ─────────────────────────────────────────────────
            feat_df   = build_features(df)
            X_raw     = feat_df.tail(1).values
            X_scaled  = scaler.transform(X_raw)

            # ── regime ───────────────────────────────────────────────────
            regime_str, gmm_state, hmm_state, gmm_probs, hmm_probs = \
                detect_regime(X_scaled, gmm, hmm, dim_scaler, dim_pca,
                              good_dims, regime_names)

            # ── UDL prediction ────────────────────────────────────────────
            buy_prob, sell_prob = predict_udl(X_raw, udl)

            # ── signal ────────────────────────────────────────────────────
            signal, confidence, reason = generate_signal(
                buy_prob, sell_prob, regime_str)

            # ── ATR / risk ────────────────────────────────────────────────
            atr_pips = float(feat_df["atr_14_pips"].iloc[-1])
            entry    = ask if signal == "BUY" else bid
            sl, tp, sl_pips = compute_sl_tp(signal, entry, atr_pips)
            lot = compute_lot_size(bal, risk_pct=1.0, sl_pips=sl_pips)

            # ── candle data for chart ─────────────────────────────────────
            chart_df = df.tail(80)
            candles  = []
            for _, row in chart_df.iterrows():
                candles.append({
                    "t": str(row["time"]),
                    "o": round(float(row["open"]),  5),
                    "h": round(float(row["high"]),  5),
                    "l": round(float(row["low"]),   5),
                    "c": round(float(row["close"]), 5),
                    "v": int(row.get("tick_volume", 0)),
                })

            # ── colour helpers ────────────────────────────────────────────
            sig_color = {"BUY": "#00e676", "SELL": "#ff5252"}.get(signal, "#888")
            reg_color = {
                "BULL_TREND" : "#00e676",
                "BEAR_TREND" : "#ff5252",
                "RANGE_CHOP" : "#ffab40",
                "ANOMALY"    : "#7c4dff",
            }.get(regime_str, "#888")

            elapsed_ms = int((time.time() - t0) * 1000)

            # ── update global state ───────────────────────────────────────
            state.update({
                "connected"      : mt5_ok,
                "last_update"    : datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "bid"            : round(bid, 5),
                "ask"            : round(ask, 5),
                "spread_pips"    : round((ask - bid) * 10000, 1),
                "signal"         : signal,
                "signal_color"   : sig_color,
                "confidence"     : round(confidence * 100, 1),
                "buy_prob"       : round(buy_prob  * 100, 1),
                "sell_prob"      : round(sell_prob * 100, 1),
                "regime"         : regime_str,
                "regime_color"   : reg_color,
                "gmm_state"      : gmm_state,
                "hmm_state"      : hmm_state,
                "gmm_probs"      : [round(p * 100, 1) for p in gmm_probs],
                "hmm_probs"      : [round(p * 100, 1) for p in hmm_probs],
                "atr_pips"       : round(atr_pips, 1),
                "sl_price"       : sl,
                "tp_price"       : tp,
                "sl_pips"        : sl_pips,
                "lot"            : lot,
                "account_balance": round(bal, 2),
                "account_equity" : round(eq,  2),
                "account_profit" : round(profit, 2),
                "open_positions" : n_pos,
                "candles"        : candles,
                "reason"         : reason,
                "inference_ms"   : elapsed_ms,
            })

            log.info("[%s] BUY=%.2f%% SELL=%.2f%% | %s | %s | %.0fms",
                     datetime.now().strftime("%H:%M:%S"),
                     buy_prob * 100, sell_prob * 100,
                     regime_str, signal, elapsed_ms)

        except Exception as e:
            err = traceback.format_exc()
            log.error("Inference error: %s", err)
            state["errors"].append(str(e))
            state["errors"] = state["errors"][-20:]

        time.sleep(10)


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EUR/USD H4 — v18.14 Live Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/lightweight-charts/4.1.1/lightweight-charts.standalone.production.js"></script>
<style>
  :root {
    --bg0:#0a0e17; --bg1:#111827; --bg2:#1a2236; --bg3:#212d45;
    --border:#2a3a5c; --text:#e2e8f0; --muted:#64748b;
    --green:#00e676; --red:#ff5252; --amber:#ffab40;
    --purple:#7c4dff; --blue:#40c4ff; --accent:#38bdf8;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg0); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; font-size:14px; }

  /* ── top bar ── */
  .topbar {
    display:flex; align-items:center; justify-content:space-between;
    padding:10px 20px; background:var(--bg1); border-bottom:1px solid var(--border);
    position:sticky; top:0; z-index:99;
  }
  .topbar .brand { font-size:16px; font-weight:700; color:var(--accent); letter-spacing:.5px; }
  .topbar .clock { font-size:12px; color:var(--muted); }
  .topbar .status { display:flex; align-items:center; gap:6px; font-size:12px; }
  .dot { width:8px; height:8px; border-radius:50%; background:var(--muted); }
  .dot.live { background:var(--green); box-shadow:0 0 6px var(--green); animation:pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  /* ── layout ── */
  .grid-main { display:grid; grid-template-columns:280px 1fr; gap:0; min-height:calc(100vh - 49px); }
  .sidebar { background:var(--bg1); border-right:1px solid var(--border); padding:16px; overflow-y:auto; display:flex; flex-direction:column; gap:12px; }
  .main { display:flex; flex-direction:column; gap:0; }

  /* ── cards ── */
  .card { background:var(--bg2); border:1px solid var(--border); border-radius:10px; padding:14px; }
  .card-title { font-size:10px; text-transform:uppercase; letter-spacing:1px; color:var(--muted); margin-bottom:10px; }

  /* ── signal card ── */
  .signal-badge {
    font-size:32px; font-weight:900; letter-spacing:2px;
    text-align:center; padding:14px; border-radius:8px;
    background:rgba(0,0,0,.3);
  }
  .conf-bar-wrap { margin-top:10px; }
  .conf-label { display:flex; justify-content:space-between; font-size:11px; color:var(--muted); margin-bottom:4px; }
  .bar-outer { background:var(--bg3); border-radius:4px; height:8px; overflow:hidden; }
  .bar-inner { height:100%; border-radius:4px; transition:width .6s ease; }

  /* ── price card ── */
  .price-row { display:flex; justify-content:space-between; align-items:baseline; }
  .bid-ask { font-size:26px; font-weight:700; }
  .spread { font-size:11px; color:var(--muted); }

  /* ── regime ── */
  .regime-name { font-size:18px; font-weight:700; text-align:center; padding:8px; }
  .prob-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:8px; }
  .prob-cell { background:var(--bg3); border-radius:6px; padding:8px; text-align:center; }
  .prob-cell .label { font-size:10px; color:var(--muted); }
  .prob-cell .val { font-size:15px; font-weight:700; }

  /* ── stat rows ── */
  .stat-row { display:flex; justify-content:space-between; align-items:center; padding:5px 0; border-bottom:1px solid var(--border); }
  .stat-row:last-child { border:none; }
  .stat-label { color:var(--muted); font-size:12px; }
  .stat-val { font-weight:600; }

  /* ── account ── */
  .acc-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
  .acc-cell { background:var(--bg3); border-radius:6px; padding:10px; text-align:center; }
  .acc-cell .label { font-size:10px; color:var(--muted); margin-bottom:4px; }
  .acc-cell .val { font-size:16px; font-weight:700; }

  /* ── chart area ── */
  .chart-header { display:flex; align-items:center; justify-content:space-between; padding:10px 16px; background:var(--bg1); border-bottom:1px solid var(--border); }
  .chart-header h3 { font-size:13px; font-weight:600; }
  #chart { flex:1; min-height:420px; }
  .chart-wrap { flex:1; display:flex; flex-direction:column; }

  /* ── bottom stats row ── */
  .bottom-stats { display:grid; grid-template-columns:repeat(4,1fr); gap:1px; background:var(--border); border-top:1px solid var(--border); }
  .bstat { background:var(--bg1); padding:12px 16px; }
  .bstat .label { font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:.5px; }
  .bstat .val   { font-size:20px; font-weight:700; margin-top:3px; }

  /* ── refresh ring ── */
  .refresh-ring {
    width:24px; height:24px; border-radius:50%;
    border:2px solid var(--border); border-top-color:var(--accent);
    animation:spin 1s linear infinite; display:none;
  }
  @keyframes spin { to{transform:rotate(360deg)} }
  .refreshing .refresh-ring { display:inline-block; }

  /* ── errors ── */
  #err-box { font-size:11px; color:var(--red); padding:6px 16px; background:rgba(255,82,82,.08); display:none; }

  @media(max-width:900px) {
    .grid-main { grid-template-columns:1fr; }
    .sidebar { border-right:none; border-bottom:1px solid var(--border); }
    .bottom-stats { grid-template-columns:repeat(2,1fr); }
  }
</style>
</head>
<body>

<!-- TOP BAR -->
<div class="topbar" id="topbar">
  <div class="brand">⚡ EUR/USD H4 — AI Trading Dashboard v18.14</div>
  <div style="display:flex;align-items:center;gap:16px;">
    <div class="refresh-ring" id="ring"></div>
    <div class="clock" id="clock">—</div>
    <div class="status">
      <div class="dot" id="status-dot"></div>
      <span id="status-txt">Connecting…</span>
    </div>
  </div>
</div>

<div id="err-box"></div>

<div class="grid-main">

  <!-- SIDEBAR -->
  <div class="sidebar">

    <!-- SIGNAL -->
    <div class="card">
      <div class="card-title">AI Signal</div>
      <div class="signal-badge" id="signal-badge" style="color:#888">—</div>
      <div style="font-size:11px;color:var(--muted);text-align:center;margin-top:6px;" id="signal-reason">—</div>
      <div class="conf-bar-wrap" style="margin-top:12px;">
        <div class="conf-label"><span>BUY Prob</span><span id="buy-pct">—</span></div>
        <div class="bar-outer"><div class="bar-inner" id="buy-bar" style="width:0%;background:var(--green)"></div></div>
      </div>
      <div class="conf-bar-wrap" style="margin-top:8px;">
        <div class="conf-label"><span>SELL Prob</span><span id="sell-pct">—</span></div>
        <div class="bar-outer"><div class="bar-inner" id="sell-bar" style="width:0%;background:var(--red)"></div></div>
      </div>
    </div>

    <!-- PRICE -->
    <div class="card">
      <div class="card-title">Live Price</div>
      <div class="price-row">
        <div><div style="font-size:10px;color:var(--muted)">BID</div><div class="bid-ask" id="bid">—</div></div>
        <div><div style="font-size:10px;color:var(--muted)">ASK</div><div class="bid-ask" id="ask">—</div></div>
      </div>
      <div class="spread" style="margin-top:6px;">Spread: <span id="spread">—</span> pips</div>
    </div>

    <!-- REGIME -->
    <div class="card">
      <div class="card-title">Market Regime</div>
      <div class="regime-name" id="regime-name" style="color:#888">—</div>
      <div class="prob-grid" id="regime-probs">
        <div class="prob-cell"><div class="label">BULL GMM</div><div class="val" id="gp0">—</div></div>
        <div class="prob-cell"><div class="label">BEAR GMM</div><div class="val" id="gp1">—</div></div>
        <div class="prob-cell"><div class="label">RANGE GMM</div><div class="val" id="gp2">—</div></div>
        <div class="prob-cell"><div class="label">ANOM GMM</div><div class="val" id="gp3">—</div></div>
      </div>
    </div>

    <!-- RISK -->
    <div class="card">
      <div class="card-title">Risk Parameters</div>
      <div class="stat-row"><span class="stat-label">ATR (pips)</span><span class="stat-val" id="atr">—</span></div>
      <div class="stat-row"><span class="stat-label">Stop Loss</span><span class="stat-val" style="color:var(--red)" id="sl">—</span></div>
      <div class="stat-row"><span class="stat-label">Take Profit</span><span class="stat-val" style="color:var(--green)" id="tp">—</span></div>
      <div class="stat-row"><span class="stat-label">SL (pips)</span><span class="stat-val" id="sl-pips">—</span></div>
      <div class="stat-row"><span class="stat-label">Lot Size</span><span class="stat-val" id="lot">—</span></div>
    </div>

    <!-- ACCOUNT -->
    <div class="card">
      <div class="card-title">Account (IC Markets Demo)</div>
      <div class="acc-grid">
        <div class="acc-cell"><div class="label">Balance</div><div class="val" id="balance">—</div></div>
        <div class="acc-cell"><div class="label">Equity</div><div class="val" id="equity">—</div></div>
        <div class="acc-cell"><div class="label">Profit</div><div class="val" id="profit">—</div></div>
        <div class="acc-cell"><div class="label">Positions</div><div class="val" id="positions">—</div></div>
      </div>
    </div>

    <!-- INFERENCE -->
    <div class="card">
      <div class="card-title">System</div>
      <div class="stat-row"><span class="stat-label">Inference</span><span class="stat-val" id="infer-ms">—</span></div>
      <div class="stat-row"><span class="stat-label">Model</span><span class="stat-val" style="color:var(--accent)">UDL+GMM+HMM</span></div>
      <div class="stat-row"><span class="stat-label">Refresh</span><span class="stat-val">10 s</span></div>
    </div>

  </div><!-- /sidebar -->

  <!-- MAIN -->
  <div class="main">

    <div class="chart-header">
      <h3>EUR/USD — H4 Candlestick Chart (Live)</h3>
      <span style="font-size:11px;color:var(--muted)" id="bar-count">— bars</span>
    </div>

    <div class="chart-wrap">
      <div id="chart"></div>
    </div>

    <div class="bottom-stats">
      <div class="bstat"><div class="label">Signal</div><div class="val" id="bs-signal" style="color:#888">—</div></div>
      <div class="bstat"><div class="label">Confidence</div><div class="val" id="bs-conf">—</div></div>
      <div class="bstat"><div class="label">Regime</div><div class="val" id="bs-regime" style="color:#888">—</div></div>
      <div class="bstat"><div class="label">Last Update</div><div class="val" style="font-size:13px" id="bs-update">—</div></div>
    </div>

  </div><!-- /main -->

</div><!-- /grid-main -->

<script>
// ── chart setup ──────────────────────────────────────────────────────────────
const chartEl = document.getElementById("chart");
const chart = LightweightCharts.createChart(chartEl, {
  width : chartEl.offsetWidth  || 900,
  height: chartEl.offsetHeight || 420,
  layout: { background:{color:"#0a0e17"}, textColor:"#e2e8f0" },
  grid  : { vertLines:{color:"#1a2236"}, horzLines:{color:"#1a2236"} },
  crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
  rightPriceScale: { borderColor:"#2a3a5c" },
  timeScale: { borderColor:"#2a3a5c", timeVisible:true, secondsVisible:false },
});

const candleSeries = chart.addCandlestickSeries({
  upColor:"#00e676", downColor:"#ff5252",
  borderUpColor:"#00e676", borderDownColor:"#ff5252",
  wickUpColor:"#00e676",   wickDownColor:"#ff5252",
});

// resize
new ResizeObserver(() => {
  chart.applyOptions({ width: chartEl.offsetWidth, height: chartEl.offsetHeight });
}).observe(chartEl);

// ── helpers ───────────────────────────────────────────────────────────────────
function el(id){ return document.getElementById(id); }

function setTxt(id, txt, color){
  const e = el(id);
  if(!e) return;
  e.textContent = txt;
  if(color) e.style.color = color;
}

function setWidth(id, pct){
  const e = el(id);
  if(e) e.style.width = Math.min(100, Math.max(0, pct)) + "%";
}

function fmt(v, dp=5){ return Number(v).toFixed(dp); }
function fmtUSD(v){ return "$" + Number(v).toLocaleString("en-US",{minimumFractionDigits:2,maximumFractionDigits:2}); }

// ── update UI from data ───────────────────────────────────────────────────────
function applyData(d){
  // status
  const live = d.connected;
  el("status-dot").className = "dot" + (live?" live":"");
  setTxt("status-txt", live ? "LIVE — IC Markets Demo" : "SIMULATION (CSV)");
  el("clock").textContent = d.last_update;

  // signal
  setTxt("signal-badge", d.signal, d.signal_color);
  el("signal-badge").style.border = "2px solid " + d.signal_color;
  setTxt("signal-reason", d.reason || "");
  setTxt("buy-pct",  d.buy_prob  + "%");
  setTxt("sell-pct", d.sell_prob + "%");
  setWidth("buy-bar",  d.buy_prob);
  setWidth("sell-bar", d.sell_prob);

  // price
  setTxt("bid", fmt(d.bid));
  setTxt("ask", fmt(d.ask));
  setTxt("spread", d.spread_pips);

  // regime
  setTxt("regime-name", d.regime, d.regime_color);
  const gp = d.gmm_probs || [0,0,0,0];
  setTxt("gp0", gp[0] + "%");
  setTxt("gp1", gp[1] + "%");
  setTxt("gp2", gp[2] + "%");
  setTxt("gp3", gp[3] + "%");

  // risk
  setTxt("atr",     d.atr_pips  + " pips");
  setTxt("sl",      fmt(d.sl_price));
  setTxt("tp",      fmt(d.tp_price));
  setTxt("sl-pips", d.sl_pips   + " pips");
  setTxt("lot",     d.lot);

  // account
  setTxt("balance",   fmtUSD(d.account_balance));
  setTxt("equity",    fmtUSD(d.account_equity));
  const profitColor = d.account_profit >= 0 ? "#00e676" : "#ff5252";
  setTxt("profit",    fmtUSD(d.account_profit), profitColor);
  setTxt("positions", d.open_positions);

  // system
  setTxt("infer-ms", d.inference_ms + " ms");

  // bottom bar
  setTxt("bs-signal", d.signal, d.signal_color);
  setTxt("bs-conf",   d.confidence + "%");
  setTxt("bs-regime", d.regime, d.regime_color);
  setTxt("bs-update", d.last_update ? d.last_update.slice(11,19) : "—");

  // candle chart
  if(d.candles && d.candles.length){
    const bars = d.candles.map(c => ({
      time: Math.floor(new Date(c.t).getTime() / 1000),
      open: c.o, high: c.h, low: c.l, close: c.c,
    }));
    // deduplicate by time
    const seen = new Set();
    const unique = bars.filter(b => { if(seen.has(b.time)) return false; seen.add(b.time); return true; });
    unique.sort((a,b)=>a.time-b.time);
    candleSeries.setData(unique);
    chart.timeScale().scrollToRealTime();
    setTxt("bar-count", unique.length + " bars");
  }

  // errors
  if(d.errors && d.errors.length){
    el("err-box").style.display = "block";
    el("err-box").textContent = "Last error: " + d.errors[d.errors.length-1];
  } else {
    el("err-box").style.display = "none";
  }
}

// ── poll /api/state every 10 s ────────────────────────────────────────────────
async function refresh(){
  el("topbar").classList.add("refreshing");
  try {
    const r = await fetch("/api/state");
    const d = await r.json();
    applyData(d);
  } catch(e){
    console.error("Fetch error:", e);
  } finally {
    el("topbar").classList.remove("refreshing");
  }
}

refresh();
setInterval(refresh, 10000);

// live clock
setInterval(() => {
  el("clock").textContent = new Date().toUTCString().slice(0,25) + " UTC";
}, 1000);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/state")
def api_state():
    return jsonify(state)


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    (ROOT / "logs").mkdir(exist_ok=True)

    # Load models
    scaler, gmm, hmm, dim_scaler, dim_pca, good_dims, regime_names, udl = load_models()

    # Connect MT5
    mt5_ok, mt5_module = mt5_connect()

    # Start background thread
    t = threading.Thread(
        target=inference_loop,
        args=(scaler, gmm, hmm, dim_scaler, dim_pca,
              good_dims, regime_names, udl, mt5_ok, mt5_module),
        daemon=True,
    )
    t.start()
    log.info("Inference thread started")

    # Run Flask
    log.info("Dashboard → http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
