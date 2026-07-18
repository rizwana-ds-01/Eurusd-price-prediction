"""
signal_engine.py  --  EUR/USD H4  v18.14
Real ensemble: CatBoost + LightGBM + XGBoost with technical regime gating.
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def build_extended_features(X_scaled, gmm_probs, hmm_probs):
    """Append 10 regime probability columns → shape (1, 113)"""
    gp = gmm_probs
    hp = hmm_probs
    extra = np.array([[
        gp[0], gp[1], gp[2], gp[3],
        hp[0], hp[1], hp[2], hp[3],
        float(np.max(gp)), float(np.max(hp))
    ]])
    return np.hstack([X_scaled, extra])


def predict_ensemble(X_scaled, X_ext, bundle):
    """
    Returns (buy_prob, sell_prob, cat_p, lgbm_p, xgb_p)
    cat uses 64 cols, lgbm/xgb use first 62.
    """
    top_k  = bundle["top_k_idx"]
    X_topk = X_ext[:, top_k]      # (1, 64)
    X_62   = X_topk[:, :62]        # (1, 62)

    cat_p  = bundle["cat_clf"].predict_proba(X_topk)[0]
    lgbm_p = bundle["lgbm_clf"].predict_proba(X_62)[0]
    xgb_p  = bundle["xgb_clf"].predict_proba(X_62)[0]

    ens = (cat_p + lgbm_p + xgb_p) / 3.0
    return float(ens[1]), float(ens[0]), cat_p, lgbm_p, xgb_p


def generate_signal(buy_prob, sell_prob, regime, bundle):
    """
    Returns (signal, confidence, reason)
    Applies regime-specific probability gates.
    """
    # Gate by regime
    regime_gates = {
        "BULL_TREND" : bundle["regime_prob_gates"].get(0, 0.55),
        "BEAR_TREND" : bundle["regime_prob_gates"].get(1, 0.54),
        "RANGE_CHOP" : bundle["regime_prob_gates"].get(2, 0.58),
        "ANOMALY"    : 0.75,   # very high bar for anomaly
    }
    gate = regime_gates.get(regime, bundle["prob_gate"])

    # ANOMALY → HOLD always
    if regime == "ANOMALY":
        return "HOLD", max(buy_prob, sell_prob), f"HOLD — ANOMALY regime (gate={gate:.2f})"

    if buy_prob >= gate:
        return ("BUY",  buy_prob,
                f"BUY prob={buy_prob:.1%} >= gate={gate:.2f} | {regime}")

    if sell_prob >= gate:
        return ("SELL", sell_prob,
                f"SELL prob={sell_prob:.1%} >= gate={gate:.2f} | {regime}")

    return ("HOLD", max(buy_prob, sell_prob),
            f"HOLD — below gate={gate:.2f} | {regime}")
