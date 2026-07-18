# EUR/USD H4 — AI Trading Dashboard v18.14

## Quick Start (Windows)

### Option A — Double-click
```
run_live.bat
```

### Option B — PowerShell
```powershell
cd "C:\...\deployment_prod"
.\start.ps1
```

### Option C — Manual
```powershell
conda activate ml_ds_env
python app\live_dashboard.py
```

Then open: **http://localhost:5000**

---

## Project Layout
```
deployment_prod/
├── app/
│   ├── live_dashboard.py    ← Flask dashboard + MT5 loop
│   ├── feature_pipeline.py  ← 103-feature builder
│   ├── regime_engine.py     ← GMM + HMM regime detection
│   ├── signal_engine.py     ← BUY / SELL / HOLD logic
│   ├── execution_engine.py  ← MT5 order placement
│   └── risk_manager.py      ← ATR-based SL/TP/lot sizing
├── models/
│   ├── B_scaled_v6_clean.pkl   ← Scaler + feature list
│   ├── D_clusters_v9_.pkl      ← GMM + HMM regime models
│   └── udl_engine_v3.pkl       ← UDL directional engine
├── config/
│   └── settings.yaml
├── data/
│   └── eurusd_4h.csv           ← Historical H4 data
├── logs/
│   └── dashboard.log
├── requirements.txt
├── run_live.bat
└── start.ps1
```

---

## MT5 Account
- **Login** : 52869629
- **Server** : ICMarketsSC-Demo
- **Symbol** : EURUSD H4

---

## Dashboard Features
| Panel | Content |
|---|---|
| AI Signal | BUY / SELL / HOLD with confidence bars |
| Live Price | Bid / Ask / Spread (real-time) |
| Market Regime | BULL_TREND / BEAR_TREND / RANGE_CHOP / ANOMALY |
| Regime Probs | GMM cluster probabilities |
| Risk | ATR, SL, TP, lot size |
| Account | Balance, equity, open P&L, positions |
| Chart | Live H4 candlestick (80 bars, auto-scroll) |
| Auto-refresh | Every **10 seconds** via AJAX |

---

## Fallback Mode
If MT5 is not installed or the connection fails, the system automatically
switches to **CSV simulation mode** using `data/eurusd_4h.csv`.
The dashboard still shows all signals and regime data.

---

## Requirements
```
pip install -r requirements.txt
```
MetaTrader5 is Windows-only. On Linux/Mac the system runs in simulation mode.
