"""
signal_engine.py  —  EUR/USD H4  v18.14
Generates BUY / SELL / HOLD signal using UDL engine predictions + regime.
"""

import numpy as np
import pickle
import joblib
import warnings
warnings.filterwarnings("ignore")


def load_udl_engine(path="models/udl_engine_v3.pkl"):
    with open(path, "rb") as f:
        udl = pickle.load(f)
    return udl


def predict_udl(X_raw, udl):
    """
    Run the UDL regression engine to get directional predictions.
    Returns (buy_prob, sell_prob)
    """
    try:
        X_s = udl["scaler_inp"].transform(X_raw)
        X_d = udl["dim_scaler"].transform(X_s[:, udl["GOOD_DIMS"]])
        X_p = udl["dim_pca"].transform(X_d)
        preds = udl["reg"].predict(X_p)[0]          # multi-output probabilities
        # preds[0] = up probability, preds[1] = down probability (clipped to [0,1])
        buy_prob  = float(np.clip(preds[0], 0, 1))
        sell_prob = float(np.clip(preds[1], 0, 1))
        return buy_prob, sell_prob
    except Exception:
        return 0.5, 0.5


def generate_signal(buy_prob, sell_prob, regime,
                    threshold=0.52, strong_threshold=0.60):
    """
    Returns: (signal_str, confidence, reason)
    """

    # Regime gating
    if regime in ("RANGE_CHOP",):
        # Range: use mean-reversion thresholds (lower bar)
        if buy_prob > threshold:
            return "BUY",  buy_prob,  "RANGE mean-reversion long"
        if sell_prob > threshold:
            return "SELL", sell_prob, "RANGE mean-reversion short"

    elif regime == "BULL_TREND":
        # Trend-following: only BUY, higher threshold
        if buy_prob > strong_threshold:
            return "BUY", buy_prob, "BULL_TREND continuation"

    elif regime == "BEAR_TREND":
        # Trend-following: only SELL
        if sell_prob > strong_threshold:
            return "SELL", sell_prob, "BEAR_TREND continuation"

    # ANOMALY or weak signal → HOLD
    return "HOLD", max(buy_prob, sell_prob), f"HOLD ({regime})"
