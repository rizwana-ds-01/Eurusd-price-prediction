"""
streamlit_app.py — v18.14  PROFESSIONAL LIGHT DASHBOARD
Clean white/slate theme · Live data · BUY + SELL balanced · Virtual P&L balance
Run: D:\conda_envs\forex_prod\python.exe -m streamlit run streamlit_app.py

v18.14 FIXES:
  ✓ Balance shows virtual_balance (paper P&L tracking), not stuck $200
  ✓ SELL signals via RANGE mean-reversion override (price vs EMA20)
  ✓ daily_loss display fixed (was multiplying by 100 twice)
  ✓ virtual_balance and mt5_balance both shown in topbar
  ✓ PnL pip + USD value shown in performance card
"""
import os, json, time
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

_DIR      = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(_DIR, "live_data.json")
SCORE_THR = 0.52

st.set_page_config(
    page_title="ICM · EUR/USD · v18.14",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════
# LIGHT THEME CSS  — clean professional white/slate
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── base ── */
*, *::before, *::after { box-sizing: border-box; }
[data-testid="stAppViewContainer"] {
    background: #F0F2F5;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"]  { background: transparent; }
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}
.block-container { padding: 0.5rem 0.9rem 1rem !important; max-width: 100% !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── top bar ── */
.topbar {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 10px 20px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
    position: relative;
    overflow: hidden;
}
.topbar::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #6366F1, #06B6D4, #10B981);
}
.tb-pair { font-size: 1.15rem; font-weight: 800; color: #0F172A; letter-spacing: 2px; }
.tb-price { font-size: 1.5rem; font-weight: 700; color: #6366F1; font-variant-numeric: tabular-nums; font-family: 'JetBrains Mono', monospace; }
.tb-lbl { font-size: .56rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; }
.tb-val { font-size: .82rem; color: #475569; font-weight: 500; }
.tb-cd  { font-size: 1.1rem; font-weight: 700; color: #6366F1; font-variant-numeric: tabular-nums; font-family: 'JetBrains Mono', monospace; }
.dot-on  { display:inline-block;width:8px;height:8px;border-radius:50%;background:#10B981;margin-right:5px;animation:pulse 1.5s infinite; }
.dot-off { display:inline-block;width:8px;height:8px;border-radius:50%;background:#EF4444;margin-right:5px; }
@keyframes pulse { 0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(16,185,129,.4)} 50%{opacity:.8;box-shadow:0 0 0 5px rgba(16,185,129,0)} }

/* ── cards ── */
.card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
.card-buy  { border-left: 4px solid #10B981; }
.card-sell { border-left: 4px solid #EF4444; }
.card-hold { border-left: 4px solid #94A3B8; }
.card-blue { border-left: 4px solid #6366F1; }
.card-amber{ border-left: 4px solid #F59E0B; }

/* ── signal box ── */
.sigbox {
    text-align: center;
    padding: 20px 10px;
    border-radius: 12px;
    margin-bottom: 10px;
}
.sigbox-buy  { background: linear-gradient(135deg,#ECFDF5,#D1FAE5); border: 1.5px solid #A7F3D0; }
.sigbox-sell { background: linear-gradient(135deg,#FFF1F2,#FFE4E6); border: 1.5px solid #FECACA; }
.sigbox-hold { background: linear-gradient(135deg,#F8FAFC,#F1F5F9); border: 1.5px solid #E2E8F0; }
.sig-buy  { font-size: 2.8rem; font-weight: 800; color: #059669; letter-spacing: 6px; }
.sig-sell { font-size: 2.8rem; font-weight: 800; color: #DC2626; letter-spacing: 6px; }
.sig-hold { font-size: 2.8rem; font-weight: 800; color: #94A3B8; letter-spacing: 6px; }

/* ── regime badges ── */
.bdg { display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:6px;font-size:.68rem;font-weight:600; }
.bdg-bull  { background:#D1FAE5; color:#065F46; }
.bdg-bear  { background:#FFE4E6; color:#991B1B; }
.bdg-range { background:#FEF3C7; color:#92400E; }
.bdg-anom  { background:#F1F5F9; color:#475569; }

/* ── gate pill ── */
.gpill-ok   { background:#D1FAE5; color:#065F46; font-size:.62rem; font-weight:700; padding:2px 8px; border-radius:4px; }
.gpill-gate { background:#FEF3C7; color:#92400E; font-size:.62rem; font-weight:700; padding:2px 8px; border-radius:4px; }

/* ── section header ── */
.shdr {
    font-size: .60rem; font-weight: 700; color: #94A3B8;
    text-transform: uppercase; letter-spacing: 1.5px;
    padding: 6px 0 5px; border-bottom: 1.5px solid #F1F5F9;
    margin: 8px 0 6px; display: flex; align-items: center; gap: 6px;
}
.shdr::before { content:''; display:inline-block; width:3px; height:10px; background:#6366F1; border-radius:2px; }

/* ── labels ── */
.lbl { font-size:.58rem; color:#94A3B8; text-transform:uppercase; letter-spacing:.8px; }
.val { font-size:.88rem; color:#0F172A; font-weight:600; }

/* ── score bar ── */
.sbar { background:#F1F5F9; border-radius:4px; height:6px; position:relative; }
.sbar-fill { border-radius:4px; height:6px; }

/* ── level grid ── */
.lgrid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px; margin:6px 0; }
.lcell { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:7px 9px; text-align:center; }
.lcell-lbl { font-size:.54rem; text-transform:uppercase; letter-spacing:.8px; color:#94A3B8; }
.lcell-val { font-size:.90rem; font-weight:700; margin-top:2px; }

/* ── UDL bars ── */
.udl-row { margin: 5px 0; }
.udl-track { background:#F1F5F9; border-radius:3px; height:8px; }
.udl-fill  { border-radius:3px; height:8px; transition: width .5s ease; }

/* ── KPI box ── */
.kpi-grid { display:grid; grid-template-columns:1fr 1fr; gap:6px; margin:6px 0; }
.kpi-cell { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:8px 10px; text-align:center; }
.kpi-lbl  { font-size:.56rem; color:#94A3B8; text-transform:uppercase; letter-spacing:.8px; }
.kpi-val  { font-size:1.15rem; font-weight:700; margin-top:2px; }

/* ── PnL big ── */
.pnl-big { font-size:2rem; font-weight:800; font-family:'JetBrains Mono',monospace; }

/* ── diagnostic bar ── */
.diag {
    background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px;
    padding:6px 14px; margin-bottom:8px;
    display:flex; flex-wrap:wrap; gap:16px; align-items:center;
    font-size:.60rem;
}
.diag-item { display:flex; flex-direction:column; }
.diag-lbl  { color:#94A3B8; font-size:.52rem; text-transform:uppercase; letter-spacing:.8px; }
.diag-val  { color:#0F172A; font-weight:600; font-size:.65rem; margin-top:1px; }

/* ── open position row ── */
.pos-row {
    background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px;
    padding:8px 12px; margin-bottom:6px;
}

/* ── closed trade row ── */
.closed-row {
    display:flex; justify-content:space-between; align-items:center;
    padding:5px 0; border-bottom:1px solid #F1F5F9; font-size:.68rem;
}

/* streamlit overrides */
[data-testid="stMetricValue"]  { font-size:1.1rem !important; font-weight:700 !important; color:#0F172A !important; }
[data-testid="stMetricLabel"]  { font-size:.60rem !important; color:#94A3B8 !important; text-transform:uppercase; }
[data-testid="stMetricDelta"]  { font-size:.62rem !important; }
button[data-baseweb="tab"]     { font-size:.70rem !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=1)
def load_live():
    try:
        with open(JSON_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_csv(name, log_dir):
    p = Path(log_dir) / f"{name}_{datetime.now().strftime('%Y%m')}.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
            df = df.sort_values("ts").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def time_ago(ts_str):
    try:
        t = pd.to_datetime(ts_str)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        s = (pd.Timestamp.now(tz="UTC") - t).total_seconds()
        if s < 60:   return f"{int(s)}s ago"
        if s < 3600: return f"{int(s//60)}m ago"
        return f"{int(s//3600)}h ago"
    except Exception:
        return "?"


def next_h4():
    now = datetime.now(timezone.utc)
    s   = ((now.hour // 4 + 1) * 4 - now.hour) * 3600 - now.minute * 60 - now.second
    s   = max(0, s) if s <= 14400 else s % 14400
    return s, f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"


def sc_col(s):
    if s >= 0.80: return "#059669"
    if s >= 0.65: return "#6366F1"
    if s >= 0.52: return "#F59E0B"
    return "#94A3B8"


def rg_badge(r):
    m = {
        "BULL_TREND": ("bdg-bull",  "▲", "BULL"),
        "BEAR_TREND": ("bdg-bear",  "▼", "BEAR"),
        "RANGE_CHOP": ("bdg-range", "⬌", "RANGE"),
        "ANOMALY":    ("bdg-anom",  "⚠", "ANOM"),
    }
    cls, ico, txt = m.get(str(r), ("bdg-anom", "?", str(r)))
    return f"<span class='bdg {cls}'>{ico} {txt}</span>"


def gate_pill(g):
    cls = "gpill-ok" if g == "ACCEPTED" else "gpill-gate"
    return f"<span class='{cls}'>{g}</span>"


def pb_light(h=None, title=None):
    """Plotly light base — NO xaxis/yaxis keys (avoids double-kwarg crash)."""
    d = dict(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFAFA",
        font=dict(color="#64748B", family="Inter, sans-serif", size=10),
        margin=dict(l=8, r=8, t=28 if title else 10, b=8),
    )
    if h:     d["height"] = h
    if title: d["title"]  = dict(text=title, font=dict(size=11, color="#475569"), x=0.02, y=0.97)
    return d


def light_ax(fig, y_range=None, x_angle=0):
    fig.update_xaxes(
        gridcolor="#F1F5F9", showgrid=True, zeroline=False,
        tickangle=x_angle, tickfont=dict(size=8, color="#94A3B8"),
        linecolor="#E2E8F0",
    )
    kw = dict(
        gridcolor="#F1F5F9", showgrid=True, zeroline=False,
        tickfont=dict(size=8, color="#94A3B8"),
        linecolor="#E2E8F0",
    )
    if y_range:
        kw["range"] = y_range
    fig.update_yaxes(**kw)


def score_bar_html(score, thr=SCORE_THR):
    pct  = min(score * 100, 100)
    col  = sc_col(score)
    ok   = score >= thr
    oc   = "#059669" if ok else "#EF4444"
    ot   = "✓ PASS" if ok else "✗ HOLD"
    tp   = thr * 100
    return f"""
<div style='margin:6px 0'>
  <div class='sbar'>
    <div class='sbar-fill' style='width:{pct:.1f}%;background:{col}'></div>
    <div style='position:absolute;top:-1px;left:{tp:.0f}%;width:2px;height:8px;background:#CBD5E1'></div>
  </div>
  <div style='display:flex;justify-content:space-between;font-size:.60rem;color:#94A3B8;margin-top:3px'>
    <span>0.00</span>
    <span style='color:{col};font-weight:700;font-family:JetBrains Mono,monospace'>{score:.4f}</span>
    <span style='color:{oc};font-weight:700'>{ot}</span>
    <span>1.00</span>
  </div>
</div>"""


def udl_bar_html(label, gmm_v, hmm_v, color):
    gw = max(0, min(float(gmm_v) * 100, 100))
    hw = max(0, min(float(hmm_v) * 100, 100))
    return f"""
<div class='udl-row'>
  <div style='display:flex;justify-content:space-between;font-size:.62rem;margin-bottom:3px'>
    <span style='color:{color};font-weight:600'>{label}</span>
    <span style='color:#64748B'>
      GMM <b style='color:{color};font-family:JetBrains Mono,monospace'>{float(gmm_v):.3f}</b>
      &nbsp; HMM <b style='color:{color};font-family:JetBrains Mono,monospace;opacity:.7'>{float(hmm_v):.3f}</b>
    </span>
  </div>
  <div class='udl-track' style='margin-bottom:2px'>
    <div class='udl-fill' style='width:{gw:.1f}%;background:{color}'></div></div>
  <div class='udl-track' style='opacity:.35'>
    <div class='udl-fill' style='width:{hw:.1f}%;background:{color}'></div></div>
</div>"""


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<div style='font-size:.95rem;font-weight:700;color:#0F172A;margin-bottom:10px'>⚙ Settings</div>",
                unsafe_allow_html=True)
    refresh = st.slider("Refresh (sec)", 1, 15, 1)
    auto    = st.checkbox("Auto-refresh", True)
    src     = st.radio("Log folder", ["demo_logs", "live_logs"], index=0)
    log_dir = Path(_DIR) / src
    st.divider()
    st.markdown("""
<div style='font-size:.62rem;color:#64748B;line-height:2.1'>
<b style='color:#0F172A'>v18.14 Engine</b><br>
Score = 0.50×ML + 0.30×Regime + 0.20×HMM<br>
<b style='color:#0F172A'>Regimes</b><br>
BULL  thr=0.55  q=1.00  sz=100%<br>
BEAR  thr=0.55  q=1.00  sz=100%<br>
RANGE thr=0.52  q=0.70  sz=90% ✅<br>
ANOM  — blocked<br>
<b style='color:#0F172A'>SELL Fix v18.14</b><br>
RANGE: price&gt;EMA20 → force SELL<br>
RANGE: price&lt;EMA20 → force BUY<br>
Block only if ML very confident (&gt;70%)<br>
<b style='color:#0F172A'>Balance Fix v18.14</b><br>
Virtual P&amp;L balance (not stuck $200)<br>
Paper trades update virtual balance<br>
<b style='color:#0F172A'>Execution</b><br>
1 trade per H4 bar · direction lock<br>
Price: 1s · ML: 10s · Trade: H4 close
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
data   = load_live()
sigs   = load_csv("signals", log_dir)
orders = load_csv("orders",  log_dir)

is_live        = data is not None
price_v        = float(data.get("price",           0))      if data else 0.0
atr_v          = float(data.get("atr",             0))      if data else 0.0
spr_v          = float(data.get("spread",          0))      if data else 0.0
bar_n          = int(data.get("bar",               0))      if data else 0
bar_date_v     = str(data.get("bar_date",          ""))     if data else ""
# v18.14: show virtual_balance (paper P&L) not stuck MT5 balance
balance_v      = float(data.get("virtual_balance", data.get("balance", 10000))) if data else 10000.0
mt5_balance_v  = float(data.get("mt5_balance",     data.get("balance", 10000))) if data else 10000.0
open_pos       = data.get("open_positions", [])              if data else []
closed_jn      = data.get("closed_trades",  [])              if data else []
last_ml        = data.get("last_ml_ts",     "")              if data else ""
inf_cnt        = int(data.get("inference_count", 0))         if data else 0
traded         = bool(data.get("traded_this_bar", False))    if data else False
sig_cnt        = data.get("signal_counts", {})               if data else {}

cd_secs, cd_str = next_h4()
cd_prog = int((1 - cd_secs / 14400) * 100)


# ══════════════════════════════════════════════════════════════════════════
# TOP BAR
# ══════════════════════════════════════════════════════════════════════════
s_dot = "<span class='dot-on'></span>" if is_live else "<span class='dot-off'></span>"
s_col = "#059669" if is_live else "#EF4444"
s_txt = f"LIVE · {time_ago(data.get('ts',''))}" if is_live else "OFFLINE"
ml_ago = time_ago(last_ml) if last_ml else "—"

st.markdown(f"""
<div class='topbar'>
  <div style='display:flex;align-items:center;gap:24px;flex-wrap:wrap'>
    <span class='tb-pair'>EUR/USD</span>
    <div><div class='tb-lbl'>Live Price</div>
      <div class='tb-price'>{price_v:.5f}</div></div>
    <div><div class='tb-lbl'>Status</div>
      <div style='font-size:.75rem;font-weight:600;color:{s_col}'>{s_dot}{s_txt}</div></div>
    <div><div class='tb-lbl'>Bar · {bar_date_v}</div>
      <div class='tb-val'>#{bar_n} · H4</div></div>
    <div><div class='tb-lbl'>ATR / Spread</div>
      <div class='tb-val' style='font-family:JetBrains Mono,monospace'>{atr_v:.1f}p / {spr_v:.1f}p</div></div>
    <div>
      <div class='tb-lbl'>Virtual P&amp;L Balance</div>
      <div class='tb-val' style='color:{"#059669" if balance_v >= mt5_balance_v else "#DC2626"}'>${balance_v:,.2f}</div>
      <div style='font-size:.50rem;color:#94A3B8'>MT5: ${mt5_balance_v:,.2f}</div>
    </div>
    <div><div class='tb-lbl'>ML Update</div>
      <div class='tb-val'>{ml_ago} · #{inf_cnt}</div></div>
    <div><div class='tb-lbl'>Bar Lock</div>
      <div style='font-size:.72rem;font-weight:600;color:{"#EF4444" if traded else "#059669"}'>
        {"🔒 Traded" if traded else "🟢 Ready"}</div></div>
  </div>
  <div style='display:flex;align-items:center;gap:20px'>
    <div style='text-align:right'>
      <div class='tb-lbl'>Next H4 Bar</div>
      <div class='tb-cd'>{cd_str}</div>
      <div style='background:#F1F5F9;border-radius:3px;height:4px;width:100px;margin-top:3px'>
        <div style='background:#6366F1;height:4px;border-radius:3px;width:{cd_prog}%'></div></div>
    </div>
    <div style='text-align:right'>
      <div class='tb-lbl'>Server UTC</div>
      <div style='font-size:.68rem;color:#64748B;font-family:JetBrains Mono,monospace'>
        {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}</div>
      <div class='tb-lbl' style='margin-top:3px'>ICM · {src.replace("_logs","").upper()}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Direction diagnostic strip
if sig_cnt:
    buys  = sig_cnt.get("BUY",  0)
    sells = sig_cnt.get("SELL", 0)
    holds = sig_cnt.get("HOLD", 0)
    tot   = max(buys + sells + holds, 1)
    b_pct = buys  / tot * 100
    s_pct = sells / tot * 100
    balance_ok = 0.15 <= (sells / max(buys+sells, 1)) <= 0.85
    bal_col = "#059669" if balance_ok else "#F59E0B"
    st.markdown(f"""
<div class='diag'>
  <div class='diag-item'><span class='diag-lbl'>BUY signals</span>
    <span class='diag-val' style='color:#059669'>{buys} ({b_pct:.0f}%)</span></div>
  <div class='diag-item'><span class='diag-lbl'>SELL signals</span>
    <span class='diag-val' style='color:#DC2626'>{sells} ({s_pct:.0f}%)</span></div>
  <div class='diag-item'><span class='diag-lbl'>Hold</span>
    <span class='diag-val'>{holds}</span></div>
  <div class='diag-item'><span class='diag-lbl'>Direction balance</span>
    <span class='diag-val' style='color:{bal_col}'>{"✓ Balanced" if balance_ok else "⚠ Check model"}</span></div>
  <div class='diag-item'><span class='diag-lbl'>Open positions</span>
    <span class='diag-val' style='color:#6366F1'>{len(open_pos)} / 3</span></div>
  <div class='diag-item'><span class='diag-lbl'>ML inferences</span>
    <span class='diag-val'>#{inf_cnt}</span></div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# 3-COLUMN LAYOUT
# ══════════════════════════════════════════════════════════════════════════
col_L, col_C, col_R = st.columns([1.1, 2.9, 1.1])


# ╔══════════════════════════════════════════════════════╗
# ║  LEFT — Signal · Score · UDL                       ║
# ╚══════════════════════════════════════════════════════╝
with col_L:
    if data:
        sn     = str(data.get("signal",    "HOLD"))
        sc     = float(data.get("score",    0))
        rg     = str(data.get("regime",     "RANGE_CHOP"))
        rg_id  = int(data.get("regime_id",  2))
        gate_v = str(data.get("gate",       "?"))
        buy_p  = float(data.get("buy_p",    0.5))
        sell_p = float(data.get("sell_p",   0.5))
        tp_v   = float(data.get("tp_pip",   0))
        sl_v   = float(data.get("sl_pip",   0))
        edge_v = float(data.get("edge",     0))
        qual   = str(data.get("quality",    "?"))
        gmm_c  = float(data.get("gmm_conf", 0))
        hmm_c  = float(data.get("hmm_conf", 0))
        sz_v   = float(data.get("size_mult",1))
        rg_thr = float(data.get("regime_threshold", SCORE_THR))
        gp     = data.get("gmm_probs", [0.25]*4)
        hp     = data.get("hmm_probs", [0.25]*4)
        hmm_r  = int(np.argmax(hp))
        rr     = round(tp_v / sl_v, 2) if sl_v > 0 else 0.0
        conf   = max(buy_p, sell_p)

        # Signal box
        bx = {"BUY":"sigbox-buy","SELL":"sigbox-sell"}.get(sn,"sigbox-hold")
        tx = {"BUY":"sig-buy",   "SELL":"sig-sell"  }.get(sn,"sig-hold")
        sc_bg = {"BUY":"#059669","SELL":"#DC2626"}.get(sn,"#94A3B8")
        st.markdown(f"""
<div class='sigbox {bx}'>
  <div class='lbl' style='color:#64748B;margin-bottom:5px'>SIGNAL</div>
  <div class='{tx}'>{sn}</div>
  <div style='margin-top:9px'>{rg_badge(rg)}</div>
  <div style='margin-top:8px;display:flex;justify-content:center;gap:8px;align-items:center'>
    <span style='font-size:.62rem;color:#64748B'>
      Q: <b style='color:#0F172A'>{qual}</b>
    </span>
    {gate_pill(gate_v)}
  </div>
</div>""", unsafe_allow_html=True)

        # Score
        st.markdown("<div class='shdr'>Composite Score</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='position:relative'>{score_bar_html(sc, rg_thr)}</div>",
                    unsafe_allow_html=True)

        # Score breakdown
        rg_q   = {0:1.00, 1:1.00, 2:0.70, 3:0.00}.get(rg_id, 0.70)
        hmm_ag = 1.0 if hmm_r == rg_id else 0.5
        cf_sc  = gmm_c * hmm_ag
        for lbl2, raw2, wt2, col2 in [
            ("ML × 0.50",  f"{conf:.4f}",  f"+{0.50*conf:.4f}", "#6366F1"),
            ("Rg × 0.30",  f"{rg_q:.2f}",  f"+{0.30*rg_q:.4f}", "#F59E0B"),
            ("HMM × 0.20", f"{cf_sc:.4f}", f"+{0.20*cf_sc:.4f}", "#06B6D4"),
        ]:
            st.markdown(f"""
<div style='display:flex;justify-content:space-between;align-items:center;
  padding:4px 10px;background:#F8FAFC;border-radius:6px;margin-bottom:3px;
  border:1px solid #F1F5F9'>
  <span style='font-size:.60rem;color:#94A3B8'>{lbl2}</span>
  <span style='font-size:.62rem;color:#64748B;font-family:JetBrains Mono,monospace'>{raw2}</span>
  <span style='font-size:.72rem;font-weight:700;color:{col2};font-family:JetBrains Mono,monospace'>{wt2}</span>
</div>""", unsafe_allow_html=True)

        # ML probabilities
        st.markdown("<div class='shdr'>ML Probabilities</div>", unsafe_allow_html=True)
        bw = buy_p * 100; sw2 = sell_p * 100
        st.markdown(f"""
<div class='card' style='padding:9px 12px'>
  <div style='display:flex;justify-content:space-between;margin-bottom:4px'>
    <span class='lbl'>BUY</span>
    <span style='font-size:.78rem;font-weight:700;color:#059669;font-family:JetBrains Mono,monospace'>{buy_p:.4f}</span>
  </div>
  <div style='background:#F1F5F9;border-radius:3px;height:6px'>
    <div style='background:#10B981;height:6px;border-radius:3px;width:{bw:.1f}%'></div></div>
  <div style='display:flex;justify-content:space-between;margin-top:7px;margin-bottom:4px'>
    <span class='lbl'>SELL</span>
    <span style='font-size:.78rem;font-weight:700;color:#DC2626;font-family:JetBrains Mono,monospace'>{sell_p:.4f}</span>
  </div>
  <div style='background:#F1F5F9;border-radius:3px;height:6px'>
    <div style='background:#EF4444;height:6px;border-radius:3px;width:{sw2:.1f}%'></div></div>
</div>""", unsafe_allow_html=True)

        # UDL bars
        st.markdown("<div class='shdr'>UDL · GMM &amp; HMM</div>", unsafe_allow_html=True)
        udl_cfg = [
            ("BULL",  gp[0], hp[0], "#059669"),
            ("BEAR",  gp[1], hp[1], "#DC2626"),
            ("RANGE", gp[2], hp[2], "#D97706"),
            ("ANOM",  gp[3], hp[3], "#94A3B8"),
        ]
        for ulbl, ugmm, uhmm, ucol in udl_cfg:
            st.markdown(udl_bar_html(ulbl, ugmm, uhmm, ucol), unsafe_allow_html=True)

        agree  = hmm_r == rg_id
        ag_col = "#059669" if agree else "#F59E0B"
        ag_txt = "✓ Agree" if agree else "≠ Diverge"
        rg_nm  = ["BULL","BEAR","RANGE","ANOM"]
        gw2 = gmm_c * 100; hw2 = hmm_c * 100
        st.markdown(f"""
<div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;
  padding:6px 10px;margin-top:5px'>
  <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:5px'>
    <span style='font-size:.60rem;color:#64748B'>GMM <b style='color:#0F172A'>{rg_nm[rg_id]}</b></span>
    <span style='font-size:.62rem;font-weight:700;color:{ag_col}'>{ag_txt}</span>
    <span style='font-size:.60rem;color:{ag_col}'>HMM <b>{rg_nm[hmm_r]}</b></span>
  </div>
  <div style='display:grid;grid-template-columns:1fr 1fr;gap:6px'>
    <div>
      <div class='lbl'>GMM Conf</div>
      <div style='font-size:.95rem;font-weight:700;color:#6366F1;font-family:JetBrains Mono,monospace'>{gmm_c:.4f}</div>
      <div style='background:#F1F5F9;border-radius:2px;height:3px;margin-top:2px'>
        <div style='background:#6366F1;height:3px;border-radius:2px;width:{gw2:.1f}%'></div></div>
    </div>
    <div>
      <div class='lbl'>HMM Conf</div>
      <div style='font-size:.95rem;font-weight:700;color:{ag_col};font-family:JetBrains Mono,monospace'>{hmm_c:.4f}</div>
      <div style='background:#F1F5F9;border-radius:2px;height:3px;margin-top:2px'>
        <div style='background:{ag_col};height:3px;border-radius:2px;width:{hw2:.1f}%'></div></div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        # Trade levels
        st.markdown("<div class='shdr'>Trade Levels</div>", unsafe_allow_html=True)
        st.markdown(f"""
<div class='lgrid'>
  <div class='lcell'><div class='lcell-lbl'>TP</div>
    <div class='lcell-val' style='color:#059669'>{tp_v:.0f}p</div></div>
  <div class='lcell'><div class='lcell-lbl'>SL</div>
    <div class='lcell-val' style='color:#DC2626'>{sl_v:.0f}p</div></div>
  <div class='lcell'><div class='lcell-lbl'>R:R</div>
    <div class='lcell-val' style='color:#D97706'>{rr}</div></div>
  <div class='lcell'><div class='lcell-lbl'>Edge</div>
    <div class='lcell-val' style='color:#475569'>{edge_v:.1f}p</div></div>
  <div class='lcell'><div class='lcell-lbl'>Size</div>
    <div class='lcell-val' style='color:#475569'>{sz_v:.0%}</div></div>
  <div class='lcell'><div class='lcell-lbl'>Thr</div>
    <div class='lcell-val' style='color:#6366F1'>{rg_thr:.2f}</div></div>
</div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
<div class='sigbox sigbox-hold' style='padding:50px 10px'>
  <div class='sig-hold'>OFFLINE</div>
  <div style='font-size:.68rem;color:#94A3B8;margin-top:10px'>
    Start the trading bot first.<br>Double-click Start Trading System.
  </div>
</div>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════╗
# ║  CENTER — UDL chart + Price chart                   ║
# ╚══════════════════════════════════════════════════════╝
with col_C:

    # UDL live chart
    if data and "gmm_probs" in data:
        gp_v = [float(x) for x in data.get("gmm_probs", [0.25]*4)]
        hp_v = [float(x) for x in data.get("hmm_probs", [0.25]*4)]
        labs = ["BULL", "BEAR", "RANGE", "ANOM"]
        uc   = ["#10B981","#EF4444","#F59E0B","#94A3B8"]

        fig_u = go.Figure()
        fig_u.add_trace(go.Bar(
            name="GMM", x=labs, y=gp_v, marker_color=uc, opacity=0.88,
            text=[f"{v:.3f}" for v in gp_v], textposition="outside",
            textfont=dict(size=9, color="#475569"),
        ))
        fig_u.add_trace(go.Bar(
            name="HMM", x=labs, y=hp_v, marker_color=uc, opacity=0.35,
            text=[f"{v:.3f}" for v in hp_v], textposition="outside",
            textfont=dict(size=8, color="#94A3B8"),
        ))
        fig_u.update_layout(
            **pb_light(h=200, title="UDL Regime Probabilities — GMM (solid) vs HMM (faded)"),
            barmode="group", showlegend=True,
            legend=dict(orientation="h", x=0.72, y=1.06,
                        font=dict(size=9, color="#64748B"), bgcolor="rgba(0,0,0,0)"),
        )
        fig_u.update_xaxes(gridcolor="#F1F5F9", tickfont=dict(size=10, color="#475569"), linecolor="#E2E8F0")
        fig_u.update_yaxes(range=[0, 1.18], gridcolor="#F1F5F9", tickfont=dict(size=8, color="#94A3B8"))
        st.markdown("<div class='shdr'>UDL Regime Probabilities — Live</div>", unsafe_allow_html=True)
        st.plotly_chart(fig_u, use_container_width=True, config={"displayModeBar": False})

    # Price / Score / GMM history
    if not sigs.empty and len(sigs) > 5:
        tail = sigs.tail(120).copy()
        has_ohlc = all(c in tail.columns for c in ["open","high","low","close","ts"])

        st.markdown("<div class='shdr'>Price Action · EMA20 · EMA50 · ML Score · GMM Regime</div>",
                    unsafe_allow_html=True)

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.54, 0.22, 0.24],
            vertical_spacing=0.025,
        )
        if has_ohlc:
            fig.add_trace(go.Candlestick(
                x=tail["ts"], open=tail["open"], high=tail["high"],
                low=tail["low"], close=tail["close"], name="EUR/USD",
                increasing_line_color="#10B981", decreasing_line_color="#EF4444",
                increasing_fillcolor="rgba(16,185,129,.7)",
                decreasing_fillcolor="rgba(239,68,68,.7)",
                line_width=1,
            ), row=1, col=1)
            e20 = tail["close"].ewm(span=20, adjust=False).mean()
            e50 = tail["close"].ewm(span=50, adjust=False).mean()
            fig.add_trace(go.Scatter(x=tail["ts"], y=e20, name="EMA20",
                line=dict(color="#F59E0B", width=1.5, dash="dot"), opacity=.9), row=1, col=1)
            fig.add_trace(go.Scatter(x=tail["ts"], y=e50, name="EMA50",
                line=dict(color="#6366F1", width=1.2, dash="dot"), opacity=.7), row=1, col=1)

        if not orders.empty and "entry" in orders.columns and "direction" in orders.columns:
            for d_, sym_, col_ in [("BUY","triangle-up","#10B981"),
                                    ("SELL","triangle-down","#EF4444")]:
                if "ts" in orders.columns:
                    sub = orders[orders["direction"] == d_].tail(20)
                    if not sub.empty:
                        fig.add_trace(go.Scatter(
                            x=sub["ts"], y=sub["entry"], mode="markers",
                            marker=dict(symbol=sym_, size=11, color=col_,
                                        line=dict(color=col_, width=1.5)),
                            name=f"{d_} entry"), row=1, col=1)

        if price_v:
            fig.add_hline(y=price_v,
                line=dict(color="rgba(99,102,241,.4)", width=1.5, dash="dot"),
                annotation_text=f"  {price_v:.5f}",
                annotation_font=dict(color="#6366F1", size=9), row=1, col=1)

        if "score" in tail.columns:
            sc_v = pd.to_numeric(tail["score"], errors="coerce").fillna(0)
            fig.add_trace(go.Bar(
                x=tail["ts"], y=sc_v,
                marker_color=[sc_col(float(v)) for v in sc_v],
                name="Score", opacity=0.85,
            ), row=2, col=1)
            fig.add_hline(y=SCORE_THR,
                line=dict(color="#CBD5E1", width=1.5, dash="dash"),
                annotation_text=f"  thr {SCORE_THR}",
                annotation_font=dict(color="#94A3B8", size=8), row=2, col=1)

        for cn, cc, cl in [("gmm_p0","#10B981","Bull"),("gmm_p1","#EF4444","Bear"),
                            ("gmm_p2","#F59E0B","Range"),("gmm_p3","#94A3B8","Anom")]:
            if cn in tail.columns:
                fig.add_trace(go.Scatter(
                    x=tail["ts"],
                    y=pd.to_numeric(tail[cn], errors="coerce").fillna(0),
                    name=cl, line=dict(color=cc, width=1.3), opacity=0.85,
                ), row=3, col=1)

        fig.update_layout(
            **pb_light(h=445),
            showlegend=True,
            legend=dict(orientation="h", x=0, y=1.01,
                        font=dict(size=8, color="#64748B"), bgcolor="rgba(0,0,0,0)",
                        itemsizing="constant"),
        )
        fig.update_xaxes(gridcolor="#F1F5F9", tickfont=dict(size=7, color="#94A3B8"),
                         rangeslider_visible=False, linecolor="#E2E8F0")
        fig.update_yaxes(gridcolor="#F1F5F9", tickfont=dict(size=7, color="#94A3B8"),
                         linecolor="#E2E8F0")
        fig.update_yaxes(range=[0, 1.05], row=2, col=1)
        fig.update_yaxes(range=[0, 1.05], row=3, col=1)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    elif data:
        # Score gauge while waiting for bars
        sc_val = float(data.get("score", 0))
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sc_val,
            title={"text": "Composite Score", "font": {"color": "#64748B", "size": 13}},
            number={"font": {"color": sc_col(sc_val), "size": 44}, "suffix": ""},
            gauge={
                "axis": {"range": [0, 1], "tickcolor": "#94A3B8", "tickfont": {"size": 10}},
                "bar":  {"color": sc_col(sc_val)},
                "bgcolor": "#F8FAFC", "bordercolor": "#E2E8F0",
                "steps": [{"range": [0, SCORE_THR], "color": "#F8FAFC"},
                           {"range": [SCORE_THR, 1], "color": "rgba(16,185,129,.08)"}],
                "threshold": {"line": {"color": "#6366F1", "width": 3},
                              "thickness": 0.8, "value": SCORE_THR},
            }))
        fig_g.update_layout(**pb_light(h=280))
        fig_g.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
        st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})
        st.info("📊 Price chart appears after a few bars are processed.")
    else:
        st.markdown("""
<div class='card' style='text-align:center;padding:80px 20px'>
  <div style='font-size:.85rem;color:#64748B'>No data — start the trading bot</div>
  <div style='font-size:.70rem;color:#94A3B8;margin-top:6px'>
    Double-click <b>Start Trading System</b> on Desktop
  </div>
</div>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════╗
# ║  RIGHT — Performance · Positions · Gates            ║
# ╚══════════════════════════════════════════════════════╝
with col_R:
    if data:
        n_t  = int(data.get("trades",      0))
        wr   = float(data.get("win_rate",   0))
        pnl  = float(data.get("total_pnl",  0))
        ppt  = float(data.get("avg_ppt",    0))
        pf   = float(data.get("pf",         0))
        # daily_loss in JSON is already a fraction (e.g. 0.002 = 0.2%)
        # DO NOT multiply by 100 here — it's already stored as fraction
        dl   = float(data.get("daily_loss", 0)) * 100   # convert fraction → percent
        ot   = len(open_pos)
        # P&L in USD for display (0.01 lot × $10/pip)
        pnl_usd = pnl * 10.0 * 0.01  # approximate: avg 0.01 lots

        pnl_col  = "#059669" if pnl >= 0 else "#DC2626"
        wr_col   = "#059669" if wr  >= 50 else "#DC2626"
        pf_col   = "#059669" if pf  >= 1.1 else "#DC2626"
        ppt_col  = "#059669" if ppt >= 0  else "#DC2626"
        dl_col   = "#DC2626" if dl  >= 2  else "#059669"
        dl_w     = min(dl / 3.0 * 100, 100)
        pnl_sign = "+" if pnl >= 0 else ""
        pnl_card = "card-buy" if pnl >= 0 else "card-sell"
        dl_card  = "card-sell" if dl >= 2 else "card-buy"
        bal_chg  = balance_v - mt5_balance_v
        bal_chg_s = f"+${bal_chg:.2f}" if bal_chg >= 0 else f"-${abs(bal_chg):.2f}"
        bal_chg_col = "#059669" if bal_chg >= 0 else "#DC2626"

        st.markdown("<div class='shdr'>Performance</div>", unsafe_allow_html=True)
        st.markdown(f"""
<div class='card {pnl_card}' style='text-align:center;padding:13px'>
  <div class='lbl'>Total P&amp;L (Closed Trades)</div>
  <div class='pnl-big' style='color:{pnl_col}'>{pnl_sign}{pnl:.1f}
    <span style='font-size:.85rem;color:#94A3B8'> pip</span>
  </div>
  <div style='font-size:.62rem;color:{ppt_col};margin-top:3px'>avg {ppt:+.1f} pip/trade</div>
  <div style='font-size:.62rem;color:{bal_chg_col};margin-top:4px;font-weight:600'>
    Account P&amp;L: {bal_chg_s} &nbsp;|&nbsp; Virtual: ${balance_v:,.2f}
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown(f"""
<div class='kpi-grid'>
  <div class='kpi-cell'>
    <div class='kpi-lbl'>Win Rate</div>
    <div class='kpi-val' style='color:{wr_col}'>{wr:.1f}%</div>
  </div>
  <div class='kpi-cell'>
    <div class='kpi-lbl'>Prof Factor</div>
    <div class='kpi-val' style='color:{pf_col}'>{min(pf,99):.2f}</div>
  </div>
  <div class='kpi-cell'>
    <div class='kpi-lbl'>Trades</div>
    <div class='kpi-val' style='color:#0F172A'>{n_t}</div>
  </div>
  <div class='kpi-cell'>
    <div class='kpi-lbl'>Open</div>
    <div class='kpi-val' style='color:#6366F1'>{ot}/3</div>
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("<div class='shdr'>Daily Risk</div>", unsafe_allow_html=True)
        st.markdown(f"""
<div class='card {dl_card}' style='padding:9px 12px'>
  <div style='display:flex;justify-content:space-between;align-items:center'>
    <span class='lbl'>Daily Loss</span>
    <span style='font-size:.9rem;font-weight:700;color:{dl_col};font-family:JetBrains Mono,monospace'>{dl:.2f}%</span>
  </div>
  <div style='background:#F1F5F9;border-radius:3px;height:7px;margin:5px 0'>
    <div style='background:{dl_col};height:7px;border-radius:3px;width:{dl_w:.1f}%'></div></div>
  <div class='lbl' style='text-align:right'>limit 3.0%</div>
</div>""", unsafe_allow_html=True)

    else:
        st.info("Start bot for performance data")

    # Open positions (from JSON)
    st.markdown("<div class='shdr'>Open Positions</div>", unsafe_allow_html=True)
    if open_pos:
        for pos in open_pos:
            d_     = str(pos.get("direction","?"))
            d_col  = "#059669" if d_ == "BUY" else "#DC2626"
            c_cls  = "card-buy" if d_ == "BUY" else "card-sell"
            ent_   = float(pos.get("entry",  0))
            sl_    = float(pos.get("sl",     0))
            tp1_   = float(pos.get("tp1",    0))
            lots_  = float(pos.get("lots",   0))
            sc__   = float(pos.get("score",  0))
            rg__   = str(pos.get("regime",   "?"))[:5]
            bars_  = int(pos.get("bars_open",0))
            pnl_u  = float(pos.get("pnl_unrealised", 0))
            pnl_uc = "#059669" if pnl_u >= 0 else "#DC2626"
            pnl_us = "+" if pnl_u >= 0 else ""
            st.markdown(f"""
<div class='pos-row' style='border-left:4px solid {d_col}'>
  <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px'>
    <span style='font-size:.80rem;font-weight:700;color:{d_col}'>{d_}</span>
    <span style='font-size:.60rem;color:#64748B'>{rg__} · {bars_}bar</span>
    <span style='font-size:.72rem;font-weight:700;color:{pnl_uc};font-family:JetBrains Mono,monospace'>
      {pnl_us}{pnl_u:.1f}p</span>
  </div>
  <div style='font-size:.60rem;color:#64748B'>
    entry <b style='color:#0F172A;font-family:JetBrains Mono,monospace'>{ent_:.5f}</b>
  </div>
  <div style='font-size:.58rem;color:#94A3B8;margin-top:1px'>
    SL <b style='color:#DC2626'>{sl_:.5f}</b> &nbsp;
    TP <b style='color:#059669'>{tp1_:.5f}</b> &nbsp;
    lots {lots_:.2f}
  </div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("<div class='card' style='text-align:center;padding:14px;"
                    "color:#94A3B8;font-size:.70rem'>No open positions</div>",
                    unsafe_allow_html=True)

    # Recent closed trades (from JSON)
    if closed_jn:
        st.markdown("<div class='shdr'>Recent Closed</div>", unsafe_allow_html=True)
        for tr in list(reversed(closed_jn))[:5]:
            d_     = str(tr.get("direction","?"))
            d_col  = "#059669" if d_ == "BUY" else "#DC2626"
            pnl_t  = float(tr.get("pnl_pips", 0))
            pnl_tc = "#059669" if pnl_t >= 0 else "#DC2626"
            pnl_ts = "+" if pnl_t >= 0 else ""
            reason = str(tr.get("reason","?"))
            st.markdown(f"""
<div class='closed-row'>
  <span style='font-weight:700;color:{d_col}'>{d_}</span>
  <span style='color:#94A3B8'>{reason}</span>
  <span style='font-weight:700;color:{pnl_tc};font-family:JetBrains Mono,monospace'>
    {pnl_ts}{pnl_t:.1f}p</span>
</div>""", unsafe_allow_html=True)

    # Gate breakdown
    if not sigs.empty and "gate" in sigs.columns:
        st.markdown("<div class='shdr'>Gate Breakdown</div>", unsafe_allow_html=True)
        gc_  = sigs["gate"].value_counts()
        tot_ = len(sigs)
        g_clr = {"ACCEPTED":"#059669","BASIC_PROB":"#94A3B8",
                  "SCORE_THRESHOLD":"#CBD5E1","COST_ATR_GATE":"#F59E0B",
                  "EDGE_BELOW_MIN":"#6366F1","VOLATILITY":"#EF4444","ANOMALY":"#EF4444"}
        for gate, cnt in gc_.items():
            pct_ = cnt / tot_ * 100
            gc__ = g_clr.get(str(gate), "#94A3B8")
            st.markdown(f"""
<div style='margin-bottom:5px'>
  <div style='display:flex;justify-content:space-between;font-size:.60rem;margin-bottom:2px'>
    <span style='color:{gc__};font-weight:600'>{gate}</span>
    <span style='color:#94A3B8'>{cnt} · {pct_:.0f}%</span>
  </div>
  <div style='background:#F1F5F9;border-radius:3px;height:4px'>
    <div style='background:{gc__};height:4px;border-radius:3px;width:{pct_:.1f}%;opacity:.8'></div>
  </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# BOTTOM TABS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💰  Performance", "📋  Signal Log", "📈  Equity Curve",
    "🔬  UDL History", "📡  Market",
])


# ── TAB 1 ─────────────────────────────────────────────────────────────────
with tab1:
    # Merge JSON closed trades + orders CSV
    all_closed = pd.DataFrame(closed_jn) if closed_jn else pd.DataFrame()
    if not orders.empty and "pnl_pips" in orders.columns and "action" in orders.columns:
        csv_closed = orders[orders["action"] == "CLOSE"]
        if not csv_closed.empty:
            all_closed = pd.concat([all_closed, csv_closed], ignore_index=True)

    if all_closed.empty and not orders.empty and "pnl_pips" in orders.columns:
        all_closed = orders

    if not all_closed.empty and "pnl_pips" in all_closed.columns:
        pnls_ = pd.to_numeric(all_closed["pnl_pips"], errors="coerce").dropna()
        pnls_ = pnls_[pnls_ != 0]
    else:
        pnls_ = pd.Series(dtype=float)

    if not pnls_.empty:
        wins_ = pnls_[pnls_ > 0]; loss_ = pnls_[pnls_ <= 0]
        n_    = len(pnls_); wr_t  = len(wins_)/n_*100
        ppt_t = pnls_.mean(); tot_t = pnls_.sum()
        pf_t  = round(wins_.sum()/abs(loss_.sum()),2) if len(loss_)>0 and loss_.sum()!=0 else 999
        dd_t  = pnls_.cumsum() - pnls_.cumsum().cummax()

        cols7 = st.columns(7)
        metrics = [
            ("Trades",      str(n_),              None),
            ("Win Rate",    f"{wr_t:.1f}%",        wr_t>=55),
            ("Avg Pip",     f"{ppt_t:+.1f}",       ppt_t>0),
            ("Total PnL",   f"{tot_t:+.0f}p",      tot_t>0),
            ("Prof Factor", f"{min(pf_t,99):.2f}", pf_t>=1.2),
            ("Best",        f"+{pnls_.max():.1f}p",None),
            ("Max DD",      f"{dd_t.min():.1f}p",  None),
        ]
        for cw, (lb, vl, ok) in zip(cols7, metrics):
            dl_ = ("✓" if ok else "⚠") if ok is not None else None
            dc_ = "normal" if (ok or ok is None) else "inverse"
            cw.metric(lb, vl, delta=dl_, delta_color=dc_)

        c1_, c2_, c3_ = st.columns([2, 2, 1])
        with c1_:
            if "reason" in all_closed.columns:
                rv_ = all_closed["reason"].dropna().value_counts().reset_index()
                rv_.columns = ["reason","count"]
                if not rv_.empty:
                    fr = px.bar(rv_, x="reason", y="count",
                                color_discrete_sequence=["#6366F1"])
                    fr.update_layout(**pb_light(h=230, title="Exit Reasons"), showlegend=False)
                    fr.update_xaxes(gridcolor="#F1F5F9", tickangle=-20, tickfont=dict(size=8, color="#94A3B8"), linecolor="#E2E8F0")
                    fr.update_yaxes(gridcolor="#F1F5F9", tickfont=dict(size=8, color="#94A3B8"), linecolor="#E2E8F0")
                    st.plotly_chart(fr, use_container_width=True, config={"displayModeBar":False})
        with c2_:
            if "direction" in all_closed.columns:
                bd_ = all_closed.copy()
                bd_["pnl_pips"] = pd.to_numeric(bd_["pnl_pips"], errors="coerce")
                bd_ = bd_[bd_["pnl_pips"].notna() & (bd_["pnl_pips"] != 0)]
                if not bd_.empty:
                    grp_ = bd_.groupby("direction")["pnl_pips"].mean().round(1).reset_index()
                    fd = go.Figure(go.Bar(
                        x=grp_["direction"], y=grp_["pnl_pips"],
                        marker_color=["#10B981" if v>0 else "#EF4444" for v in grp_["pnl_pips"]]))
                    fd.update_layout(**pb_light(h=230, title="Avg PnL by Direction"), showlegend=False)
                    fd.update_xaxes(gridcolor="#F1F5F9", tickfont=dict(size=9, color="#94A3B8"), linecolor="#E2E8F0")
                    fd.update_yaxes(gridcolor="#F1F5F9", tickfont=dict(size=9, color="#94A3B8"), linecolor="#E2E8F0")
                    st.plotly_chart(fd, use_container_width=True, config={"displayModeBar":False})
        with c3_:
            fw = go.Figure(go.Pie(
                labels=["Win","Loss"], values=[len(wins_), max(len(loss_),0)],
                hole=0.6, marker=dict(colors=["#10B981","#EF4444"]), textinfo="none"))
            fw.add_annotation(text=f"{wr_t:.0f}%", x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="#0F172A", family="Inter"))
            fw.update_layout(**pb_light(h=230, title="Win Rate"), showlegend=False)
            fw.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
            st.plotly_chart(fw, use_container_width=True, config={"displayModeBar":False})
    else:
        st.info("⏳ No closed trades yet. Trades execute on H4 bar close (every 4 hours).")
        st.markdown("""
<div class='card card-blue' style='padding:12px 16px'>
  <div style='font-size:.70rem;color:#475569;line-height:2'>
    <b style='color:#0F172A'>How trades close:</b><br>
    • TP1 hit → 70% closed, SL moved to breakeven<br>
    • Runner TP hit → remaining 30% closed<br>
    • SL hit → full close at loss<br>
    • Signal flip (BUY→SELL or SELL→BUY) → early exit<br>
    • ANOMALY regime → emergency exit<br>
    • Max 12 bars (48h) → time exit
  </div>
</div>""", unsafe_allow_html=True)


# ── TAB 2 ─────────────────────────────────────────────────────────────────
with tab2:
    if not sigs.empty:
        sc2 = [c for c in ["ts","bar_date","signal","score","regime","buy_p","sell_p",
                            "edge","atr","spread","gmm_conf","hmm_conf","tp_pip","sl_pip",
                            "quality","gate","price","open_trades"] if c in sigs.columns]
        rec = (sigs.tail(200)[sc2].sort_values("ts", ascending=False)
               if "ts" in sc2 else sigs.tail(200)[sc2])
        def _cs(v): return ("color:#059669;font-weight:700" if v=="BUY" else
                            "color:#DC2626;font-weight:700" if v=="SELL" else "color:#94A3B8")
        def _cr(v): v=str(v); return ("color:#059669" if "BULL" in v else
                                       "color:#DC2626" if "BEAR" in v else
                                       "color:#D97706" if "RANGE" in v else "color:#94A3B8")
        def _cg(v): return "color:#059669;font-weight:700" if v=="ACCEPTED" else "color:#94A3B8"
        def _ck(v):
            try:    return f"color:{sc_col(float(v))};font-weight:700"
            except: return "color:#94A3B8"
        styled = rec.style
        if "signal" in sc2: styled = styled.map(_cs, subset=["signal"])
        if "regime" in sc2: styled = styled.map(_cr, subset=["regime"])
        if "gate"   in sc2: styled = styled.map(_cg, subset=["gate"])
        if "score"  in sc2:
            styled = styled.map(_ck, subset=["score"])
            styled = styled.format({"score":"{:.4f}"}, subset=["score"])
        for fc in ["buy_p","sell_p","gmm_conf","hmm_conf"]:
            if fc in sc2: styled = styled.format({fc:"{:.4f}"})
        st.dataframe(styled, use_container_width=True, height=460)
    else:
        st.info(f"Signal log: `{log_dir}/signals_YYYYMM.csv` — builds as bars are processed.")


# ── TAB 3 ─────────────────────────────────────────────────────────────────
with tab3:
    pnls3 = pd.Series(dtype=float)
    if closed_jn:
        pnls3 = pd.to_numeric(pd.DataFrame(closed_jn).get("pnl_pips", pd.Series()),
                               errors="coerce").dropna()
    if pnls3.empty and not orders.empty and "pnl_pips" in orders.columns:
        pnls3 = pd.to_numeric(orders["pnl_pips"], errors="coerce").dropna()
    pnls3 = pnls3[pnls3 != 0]

    if not pnls3.empty:
        cum3 = pnls3.cumsum(); dd3 = cum3 - cum3.cummax()
        fig_eq = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[0.65, 0.35], vertical_spacing=0.03)
        fig_eq.add_trace(go.Scatter(y=cum3.values, mode="lines",
            line=dict(color="#10B981", width=2.5), fill="tozeroy",
            fillcolor="rgba(16,185,129,.07)", name="Cumul PnL"), row=1, col=1)
        fig_eq.add_hline(y=0, line=dict(color="#CBD5E1", width=1, dash="dot"), row=1, col=1)
        fig_eq.add_trace(go.Scatter(y=dd3.values, mode="lines", fill="tozeroy",
            fillcolor="rgba(239,68,68,.08)", line=dict(color="#EF4444", width=1.5),
            name="Drawdown"), row=2, col=1)
        fig_eq.update_layout(**pb_light(h=380), showlegend=True,
            legend=dict(orientation="h", x=0, y=1.04,
                        font=dict(size=9, color="#64748B"), bgcolor="rgba(0,0,0,0)"))
        fig_eq.update_xaxes(gridcolor="#F1F5F9", tickfont=dict(size=8, color="#94A3B8"), linecolor="#E2E8F0")
        fig_eq.update_yaxes(gridcolor="#F1F5F9", tickfont=dict(size=8, color="#94A3B8"), linecolor="#E2E8F0")
        fig_eq.update_yaxes(title_text="Pip",    row=1, col=1, title_font=dict(size=9, color="#94A3B8"))
        fig_eq.update_yaxes(title_text="DD pip", row=2, col=1, title_font=dict(size=9, color="#94A3B8"))
        st.plotly_chart(fig_eq, use_container_width=True, config={"displayModeBar":False})

        fh = go.Figure(go.Histogram(x=pnls3, nbinsx=30,
            marker=dict(color=["#10B981" if v>0 else "#EF4444" for v in pnls3], opacity=0.8)))
        fh.add_vline(x=0, line=dict(color="#CBD5E1", width=2, dash="dash"))
        fh.update_layout(**pb_light(h=210, title="PnL Distribution (pip)"), showlegend=False)
        fh.update_xaxes(gridcolor="#F1F5F9", tickfont=dict(size=8, color="#94A3B8"), linecolor="#E2E8F0")
        fh.update_yaxes(gridcolor="#F1F5F9", tickfont=dict(size=8, color="#94A3B8"), linecolor="#E2E8F0")
        st.plotly_chart(fh, use_container_width=True, config={"displayModeBar":False})
    else:
        st.info("⏳ Equity curve builds as trades close.")


# ── TAB 4 ─────────────────────────────────────────────────────────────────
with tab4:
    if not sigs.empty and len(sigs) > 5:
        t200 = sigs.tail(200)
        c1t, c2t = st.columns(2)
        with c1t:
            uc4 = [c for c in ["gmm_p0","gmm_p1","gmm_p2","gmm_p3"] if c in t200.columns]
            if uc4 and "ts" in t200.columns:
                fu4 = go.Figure()
                for cn4, cc4, cl4 in zip(uc4,
                    ["#10B981","#EF4444","#F59E0B","#94A3B8"],
                    ["GMM Bull","GMM Bear","GMM Range","GMM Anom"]):
                    fu4.add_trace(go.Scatter(x=t200["ts"],
                        y=pd.to_numeric(t200[cn4], errors="coerce"),
                        name=cl4, line=dict(color=cc4, width=1.8)))
                fu4.update_layout(**pb_light(h=275, title="GMM Regime Probabilities — History"),
                    legend=dict(font=dict(size=8, color="#64748B"), bgcolor="rgba(0,0,0,0)"))
                fu4.update_xaxes(gridcolor="#F1F5F9", tickfont=dict(size=7, color="#94A3B8"), linecolor="#E2E8F0")
                fu4.update_yaxes(range=[0,1.05], gridcolor="#F1F5F9", tickfont=dict(size=7, color="#94A3B8"), linecolor="#E2E8F0")
                st.plotly_chart(fu4, use_container_width=True, config={"displayModeBar":False})
        with c2t:
            if "gmm_conf" in t200.columns and "ts" in t200.columns:
                fc4 = go.Figure()
                fc4.add_trace(go.Scatter(x=t200["ts"],
                    y=pd.to_numeric(t200["gmm_conf"], errors="coerce"),
                    name="GMM Conf", line=dict(color="#6366F1", width=2)))
                if "hmm_conf" in t200.columns:
                    fc4.add_trace(go.Scatter(x=t200["ts"],
                        y=pd.to_numeric(t200["hmm_conf"], errors="coerce"),
                        name="HMM Conf", line=dict(color="#F59E0B", width=1.8, dash="dot")))
                fc4.update_layout(**pb_light(h=275, title="Confidence History"),
                    legend=dict(font=dict(size=8, color="#64748B"), bgcolor="rgba(0,0,0,0)"))
                fc4.update_xaxes(gridcolor="#F1F5F9", tickfont=dict(size=7, color="#94A3B8"), linecolor="#E2E8F0")
                fc4.update_yaxes(range=[0,1.05], gridcolor="#F1F5F9", tickfont=dict(size=7, color="#94A3B8"), linecolor="#E2E8F0")
                st.plotly_chart(fc4, use_container_width=True, config={"displayModeBar":False})
        if "regime" in sigs.columns:
            c1p, c2p = st.columns(2)
            with c1p:
                rv4 = sigs["regime"].value_counts().reset_index(); rv4.columns=["Regime","Bars"]
                frp = go.Figure(go.Pie(labels=rv4["Regime"], values=rv4["Bars"], hole=0.55,
                    marker=dict(colors=["#10B981","#EF4444","#F59E0B","#94A3B8"]),
                    textinfo="label+percent", textfont=dict(size=10, color="#475569")))
                frp.update_layout(**pb_light(h=260, title="Regime Distribution"), showlegend=False)
                frp.update_layout(paper_bgcolor="#FFFFFF")
                st.plotly_chart(frp, use_container_width=True, config={"displayModeBar":False})
            with c2p:
                if "gate" in sigs.columns:
                    gv4 = sigs["gate"].value_counts().reset_index(); gv4.columns=["Gate","Count"]
                    fgv = px.bar(gv4, x="Gate", y="Count", color="Gate",
                        color_discrete_map={"ACCEPTED":"#10B981","ANOMALY":"#EF4444",
                            "VOLATILITY":"#EF4444","BASIC_PROB":"#94A3B8",
                            "SCORE_THRESHOLD":"#CBD5E1","COST_ATR_GATE":"#F59E0B",
                            "EDGE_BELOW_MIN":"#6366F1"})
                    fgv.update_layout(**pb_light(h=260, title="Gate Counts"), showlegend=False)
                    fgv.update_xaxes(gridcolor="#F1F5F9", tickangle=-25, tickfont=dict(size=8, color="#94A3B8"), linecolor="#E2E8F0")
                    fgv.update_yaxes(gridcolor="#F1F5F9", tickfont=dict(size=8, color="#94A3B8"), linecolor="#E2E8F0")
                    st.plotly_chart(fgv, use_container_width=True, config={"displayModeBar":False})
    else:
        st.info("UDL history builds as bars are processed.")


# ── TAB 5 ─────────────────────────────────────────────────────────────────
with tab5:
    if not sigs.empty and len(sigs) > 5:
        t100 = sigs.tail(100)
        c1m, c2m = st.columns(2)
        with c1m:
            if "atr" in t100.columns and "ts" in t100.columns:
                av5 = pd.to_numeric(t100["atr"], errors="coerce")
                fa5 = go.Figure(go.Scatter(x=t100["ts"], y=av5, fill="tozeroy",
                    fillcolor="rgba(245,158,11,.08)", line=dict(color="#F59E0B", width=2), name="ATR"))
                fa5.add_hline(y=8,  line=dict(color="#EF4444", dash="dot"),
                    annotation_text=" floor 8", annotation_font=dict(size=8, color="#EF4444"))
                fa5.add_hline(y=55, line=dict(color="#EF4444", dash="dot"),
                    annotation_text=" ceil 55", annotation_font=dict(size=8, color="#EF4444"))
                fa5.update_layout(**pb_light(h=255, title="ATR (pip) — tradeable 8-55"), showlegend=False)
                fa5.update_xaxes(gridcolor="#F1F5F9", tickfont=dict(size=7, color="#94A3B8"), linecolor="#E2E8F0")
                fa5.update_yaxes(gridcolor="#F1F5F9", tickfont=dict(size=7, color="#94A3B8"), linecolor="#E2E8F0")
                st.plotly_chart(fa5, use_container_width=True, config={"displayModeBar":False})
                ca = float(av5.iloc[-1]) if len(av5)>0 else 0
                st.caption(f"ATR: {ca:.1f}pip · {'✅ tradeable' if 8<=ca<=55 else '⚠️ outside range'}")
        with c2m:
            if "spread" in t100.columns and "ts" in t100.columns:
                sv5 = pd.to_numeric(t100["spread"], errors="coerce")
                fs5 = go.Figure(go.Scatter(x=t100["ts"], y=sv5, fill="tozeroy",
                    fillcolor="rgba(99,102,241,.07)", line=dict(color="#6366F1", width=1.8), name="Spread"))
                fs5.update_layout(**pb_light(h=255, title="Spread (pip)"), showlegend=False)
                fs5.update_xaxes(gridcolor="#F1F5F9", tickfont=dict(size=7, color="#94A3B8"), linecolor="#E2E8F0")
                fs5.update_yaxes(gridcolor="#F1F5F9", tickfont=dict(size=7, color="#94A3B8"), linecolor="#E2E8F0")
                st.plotly_chart(fs5, use_container_width=True, config={"displayModeBar":False})
                cs = float(sv5.iloc[-1]) if len(sv5)>0 else 0
                st.caption(f"Spread: {cs:.2f}pip")
    else:
        st.info("Market data builds as bars are processed.")


# ══════════════════════════════════════════════════════════════════════════
# FOOTER + AUTO-REFRESH
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='text-align:center;color:#CBD5E1;font-size:.58rem;padding:8px 0;margin-top:6px;
  border-top:1px solid #E2E8F0;font-family:Inter,sans-serif'>
  v18.14 · EUR/USD H4 · CatBoost + GMM + HMM · RANGE MR-override · True BUY+SELL ·
  Virtual balance tracking · 1s price · 10s ML · {datetime.now():%Y-%m-%d %H:%M:%S}
</div>""", unsafe_allow_html=True)

if auto:
    time.sleep(refresh)
    st.rerun()
else:
    if st.button("🔄  Refresh Now"):
        st.rerun()