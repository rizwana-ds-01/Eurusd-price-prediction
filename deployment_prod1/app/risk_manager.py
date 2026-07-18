"""
risk_manager.py  —  EUR/USD H4  v18.14
ATR-based dynamic TP/SL and position sizing.
"""

import numpy as np


def compute_sl_tp(signal, entry_price, atr_pips,
                  sl_mult=1.5, tp_mult=2.5, pip=0.0001):
    """
    Returns (stop_loss_price, take_profit_price, risk_pips)
    """
    sl_pips = atr_pips * sl_mult
    tp_pips = atr_pips * tp_mult

    if signal == "BUY":
        sl = entry_price - sl_pips * pip
        tp = entry_price + tp_pips * pip
    elif signal == "SELL":
        sl = entry_price + sl_pips * pip
        tp = entry_price - tp_pips * pip
    else:
        sl = tp = entry_price

    return round(sl, 5), round(tp, 5), round(sl_pips, 1)


def compute_lot_size(account_balance, risk_pct=1.0, sl_pips=20.0,
                     pip_value=10.0, min_lot=0.01, max_lot=1.0):
    """
    Standard fixed-fractional lot sizing.
    pip_value = USD per pip per standard lot (EURUSD ≈ $10).
    """
    risk_usd = account_balance * (risk_pct / 100.0)
    lot = risk_usd / (sl_pips * pip_value)
    lot = max(min_lot, min(max_lot, round(lot, 2)))
    return lot
