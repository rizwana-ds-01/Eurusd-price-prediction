"""
udl_engine.py — v18.7 Real UDL (Ridge→GMM→HMM)
Returns (regime, gmm_probs, hmm_probs, size_mult) to live_pipeline.
"""
import os, sys, numpy as np, pickle, logging
from collections import deque
from typing import Tuple

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path: sys.path.insert(0, _DIR)
logger = logging.getLogger(__name__)

REGIME_NAMES = {0:'BULL_TREND',1:'BEAR_TREND',2:'RANGE_CHOP',3:'ANOMALY'}

class UDLEngine:
    def __init__(self, udl_path: str):
        with open(udl_path,'rb') as f: pkg=pickle.load(f)
        self.scaler_inp = pkg['scaler_inp']
        self.reg        = pkg['reg']
        self.dim_scaler = pkg['dim_scaler']
        self.dim_pca    = pkg['dim_pca']
        self.gmm        = pkg['gmm']
        self.hmm        = pkg['hmm']
        self.GOOD_DIMS  = pkg['GOOD_DIMS']
        self.hmm_window = pkg.get('hmm_window',100)
        seed            = pkg.get('hmm_seed_buffer',[])
        self._pca4_buf  = deque([np.array(s,dtype=np.float64) for s in seed],
                                maxlen=self.hmm_window)
        logger.info(f"UDLEngine | GOOD_DIMS={self.GOOD_DIMS} | "
                    f"seed_buffer={len(self._pca4_buf)} bars")

    def predict(self, X_scaled: np.ndarray) -> Tuple[int,np.ndarray,np.ndarray,float]:
        if X_scaled.ndim==1: X_scaled=X_scaled.reshape(1,-1)
        assert X_scaled.shape[1]==103, f"Expected 103 features, got {X_scaled.shape[1]}"
        X_inp  = self.scaler_inp.transform(X_scaled)
        X_emb  = self.reg.predict(X_inp).astype(np.float32)
        X_good = X_emb[:, self.GOOD_DIMS]
        X_sc   = self.dim_scaler.transform(X_good)
        X_pca4 = self.dim_pca.transform(X_sc)
        gmm_p  = self.gmm.predict_proba(X_pca4)[0]
        regime = int(gmm_p.argmax())
        self._pca4_buf.append(X_pca4[0].astype(np.float64))
        if len(self._pca4_buf) >= 10:
            hmm_p = self.hmm.predict_proba(np.array(self._pca4_buf))[-1]
        else:
            hmm_p = gmm_p.copy()
        conf      = float(gmm_p.max())
        size_mult = float(np.clip(0.6 + conf*0.8, 0.5, 1.4))
        if regime == 3: size_mult = 0.5
        return regime, gmm_p, hmm_p, size_mult

    def get_regime_name(self, r: int) -> str:
        return REGIME_NAMES.get(r, f'UNKNOWN_{r}')