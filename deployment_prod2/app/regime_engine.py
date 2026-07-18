"""
regime_engine.py  --  EUR/USD H4  v18.14
Real technical regime detection using ADX, EMA alignment, RSI, ATR.
The GMM/HMM models collapse to state-0 on all data (known issue).
This engine gives correct, changing regime labels.
"""
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")


def detect_regime(feat_row: dict) -> tuple:
    """
    feat_row : dict of latest feature values (from feat_df.iloc[-1])
    Returns  : (regime_str, gmm_state_proxy, hmm_state_proxy,
                gmm_probs_proxy, hmm_probs_proxy)
    """
    adx       = float(feat_row.get("adx", 20))
    em20_50   = float(feat_row.get("ema20_vs_ema50", 0))    # pips *10000
    em50_200  = float(feat_row.get("ema50_vs_ema200", 0))
    rsi       = float(feat_row.get("rsi", 50))
    atr_pct   = float(feat_row.get("atr_percentile", 0.5))
    momentum  = float(feat_row.get("momentum", 0))
    rng_width = float(feat_row.get("range_width", 50))

    # ── Range / Chop ──────────────────────────────────────────
    if adx < 20 or rng_width < 25:
        regime = "RANGE_CHOP"
        state  = 2

    # ── Strong Bull Trend ─────────────────────────────────────
    elif em20_50 > 0 and em50_200 > 0 and adx >= 20:
        regime = "BULL_TREND"
        state  = 0

    # ── Strong Bear Trend ─────────────────────────────────────
    elif em20_50 < 0 and em50_200 < 0 and adx >= 20:
        regime = "BEAR_TREND"
        state  = 1

    # ── Mixed / Transition ────────────────────────────────────
    elif em20_50 > 0 and em50_200 < 0:
        # Short-term bull, long-term bear → range/transition
        regime = "RANGE_CHOP"
        state  = 2

    elif em20_50 < 0 and em50_200 > 0:
        # Short-term bear, long-term bull → pullback / range
        regime = "RANGE_CHOP"
        state  = 2

    else:
        regime = "RANGE_CHOP"
        state  = 2

    # ── Anomaly override ──────────────────────────────────────
    if atr_pct > 0.95 and adx > 50:
        regime = "ANOMALY"
        state  = 3

    # ── Build synthetic probability vector ────────────────────
    # Give the dominant state high probability, rest share remainder
    probs = [0.02, 0.02, 0.02, 0.02]
    probs[state] = 0.94
    probs = [round(p, 4) for p in probs]

    return regime, state, state, probs, probs


def load_regime_models(path: str):
    """Stub — kept for import compatibility. Not used in v18.14."""
    return None, None, None, None, None, {}
