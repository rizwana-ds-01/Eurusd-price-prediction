"""
regime_engine.py  —  EUR/USD H4  v18.14
Uses GMM + HMM from D_clusters_v9 to detect market regime.
"""

import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")


def load_regime_models(bundle_path="models/D_clusters_v9_.pkl"):
    with open(bundle_path, "rb") as f:
        dc = pickle.load(f)
    gmm         = dc["gmm"]
    hmm         = dc["hmm"]
    dim_scaler  = dc["dim_scaler"]
    dim_pca     = dc["dim_pca"]
    good_dims   = dc["GOOD_DIMS"]
    regime_names= dc["REGIME_NAMES"]          # {0:'BULL_TREND',1:'BEAR_TREND',2:'RANGE_CHOP',3:'ANOMALY'}
    return gmm, hmm, dim_scaler, dim_pca, good_dims, regime_names


def detect_regime(X_scaled, gmm, hmm, dim_scaler, dim_pca, good_dims, regime_names):
    """
    X_scaled : 2-D array, shape (1, n_features)  — already scaler-transformed
    Returns  : (regime_str, gmm_state, hmm_state, gmm_probs, hmm_probs)
    """
    try:
        # Project to regime PCA space
        X_dim = dim_scaler.transform(X_scaled[:, good_dims])
        X_pca = dim_pca.transform(X_dim)

        # GMM
        gmm_state = int(gmm.predict(X_pca)[0])
        gmm_probs = gmm.predict_proba(X_pca)[0]

        # HMM
        hmm_state = int(hmm.predict(X_pca)[0])
        hmm_probs = hmm.predict_proba(X_pca)[0]

        regime_str = regime_names.get(gmm_state, "UNKNOWN")
        return regime_str, gmm_state, hmm_state, gmm_probs, hmm_probs

    except Exception as e:
        return "UNKNOWN", -1, -1, np.zeros(4), np.zeros(4)
