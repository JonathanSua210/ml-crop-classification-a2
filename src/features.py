"""Turn a (T, 13) Sentinel-2 time series into a fixed-length feature vector."""
import numpy as np

BAND_NAMES = [f"b{i}" for i in range(13)]
STATS = ["mean", "std", "min", "max"]

FEATURE_NAMES = [f"{b}_{s}" for b in BAND_NAMES for s in STATS]


def extract_features(x):
    """x: numpy array or tensor of shape (T, 13). Returns 1D array of length 52."""
    x = np.asarray(x)
    feats = np.concatenate([
        x.mean(axis=0),
        x.std(axis=0),
        x.min(axis=0),
        x.max(axis=0),
    ])
    return feats
