"""
signal_engine.py  —  EUR/USD H4  v18.14
Uses deploy_ml_bundle_v18_7 (CatBoost + LightGBM + XGBoost ensemble).
Fixed: was incorrectly passing PCA-reduced data to the regressor.
Now uses the correct 3-model ensemble with proper feature selection.
"""

import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")


def load_signal_bundle(path="models/deploy_ml_bundle_v18_7.pkl"):
    with open(path, "rb") as f:
        bun = pickle.load(f)
    return bun


def predict_signal(X_scaled, bun, feat_df_columns):
    """
    X_scaled : (1, 103) — already scaler-transformed
    bun       : deploy_ml_bundle_v18_7
    feat_df_columns: list of 103 feature names (from feat_df.columns)

    Returns (up_prob, down_prob)
    """
    try:
        feat_sel_names = bun["feat_sel"]  # 64 names

        # Build X_64 by matching feature names
        X_64 = np.zeros((1, 64))
        for i, name in enumerate(feat_sel_names):
            if name in feat_df_columns:
                X_64[0, i] = X_scaled[0, feat_df_columns.index(name)]

        # lgbm and xgb use first 62 columns
        X_62 = X_64[:, :62]

        cat_p  = bun["cat_clf"].predict_proba(X_64)
        lgbm_p = bun["lgbm_clf"].predict_proba(X_62)
        xgb_p  = bun["xgb_clf"].predict_proba(X_62)

        up_prob   = float((cat_p[0][1] + lgbm_p[0][1] + xgb_p[0][1]) / 3.0)
        down_prob = 1.0 - up_prob
        return up_prob, down_prob

    except Exception as e:
        return 0.5, 0.5


def generate_signal(up_prob, down_prob, regime, gmm_state, bun):
    """
    Returns (signal_str, confidence, reason)
    Uses regime-specific probability gates from the bundle.
    """
    # regime-gated threshold
    regime_gates = bun.get("regime_prob_gates", {})
    default_gate = bun.get("prob_gate", 0.53)
    thresh = regime_gates.get(gmm_state, default_gate)

    if up_prob >= thresh:
        return "BUY",  round(up_prob,   4), f"{regime} | gate={thresh:.2f}"
    elif down_prob >= thresh:
        return "SELL", round(down_prob, 4), f"{regime} | gate={thresh:.2f}"
    else:
        return "HOLD", round(max(up_prob, down_prob), 4), f"HOLD ({regime}) | gate={thresh:.2f}"
