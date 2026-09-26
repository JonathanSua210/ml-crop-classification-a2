"""
Feature extraction, v2.

The v1 features (per-band mean/std/min/max) are ORDER-FREE: permuting the 45
acquisition dates leaves them unchanged. That makes them a clean baseline but
it means the model cannot represent phenology at all - and phenology timing,
not spectral magnitude, is what actually separates wheat from barley.

v2 adds three order-aware families:

  segment means   coarse temporal shape; 6 evenly spaced windows per band
  harmonics       smooth seasonal shape; order-2 Fourier fit per band
  NDVI phenology  task-specific curve descriptors (peak height, peak timing,
                  green-up and senescence slopes, season integral)

Naming convention lets downstream scripts select families by prefix:
  b{i}_mean / _std / _min / _max      v1 basic          52
  b{i}_seg{j}                          segment means     78
  b{i}_h0 / _h1c / _h1s / _h2c / _h2s  harmonics         65
  ndvi_*                               phenology          7
                                                        ---
                                                        202

BAND ORDER ASSUMPTION
---------------------
BreizhCrops L1C is assumed to carry the 13 Sentinel-2 bands in standard
order, so index 3 is B4 (red, 665nm) and index 7 is B8 (NIR, 842nm).
This is an assumption, not something I read off the data, so
`sanity_check_ndvi` exists to falsify it: vegetated parcels must show
clearly positive NDVI peaking mid-season. If the printed values are near
zero, negative, or peak in winter, the indices are wrong.
"""
import numpy as np

RED_IDX = 3
NIR_IDX = 7

N_SEGMENTS = 6
N_HARMONICS = 2
EPS = 1e-8

BAND_NAMES = [f"b{i}" for i in range(13)]

# ---------------------------------------------------------------- names
BASIC_NAMES = [f"{b}_{s}" for b in BAND_NAMES for s in ("mean", "std", "min", "max")]
SEGMENT_NAMES = [f"{b}_seg{j}" for b in BAND_NAMES for j in range(N_SEGMENTS)]
HARMONIC_NAMES = [f"{b}_{h}" for b in BAND_NAMES
                  for h in ("h0", "h1c", "h1s", "h2c", "h2s")]
NDVI_NAMES = ["ndvi_peak", "ndvi_argmax", "ndvi_mean", "ndvi_integral",
              "ndvi_greenup", "ndvi_senescence", "ndvi_amplitude"]

FEATURE_NAMES_V2 = BASIC_NAMES + SEGMENT_NAMES + HARMONIC_NAMES + NDVI_NAMES

FAMILIES = {
    "basic": BASIC_NAMES,
    "segment": SEGMENT_NAMES,
    "harmonic": HARMONIC_NAMES,
    "ndvi": NDVI_NAMES,
}


def _design_matrix(T):
    """Order-2 Fourier design matrix over a season of length T: [1, cos, sin, cos2, sin2]."""
    t = np.arange(T, dtype=np.float64) / T
    cols = [np.ones(T)]
    for k in range(1, N_HARMONICS + 1):
        cols.append(np.cos(2 * np.pi * k * t))
        cols.append(np.sin(2 * np.pi * k * t))
    return np.stack(cols, axis=1)          # (T, 5)


_CACHE = {}


def _pinv_for(T):
    """Cache the pseudo-inverse: the design matrix is identical for every parcel."""
    if T not in _CACHE:
        _CACHE[T] = np.linalg.pinv(_design_matrix(T))   # (5, T)
    return _CACHE[T]


def extract_features_v2(x):
    """
    x : (T, 13) array of reflectance, T acquisitions by 13 bands.
    Returns a 1D float64 array of length len(FEATURE_NAMES_V2).
    """
    x = np.asarray(x, dtype=np.float64)
    T, B = x.shape

    # --- basic (order free) ---
    basic = np.concatenate([x.mean(0), x.std(0), x.min(0), x.max(0)])

    # --- segment means (coarse order) ---
    edges = np.linspace(0, T, N_SEGMENTS + 1).astype(int)
    seg = np.stack([x[edges[j]:edges[j + 1]].mean(0) for j in range(N_SEGMENTS)],
                   axis=1)                                   # (13, N_SEGMENTS)
    segment = seg.reshape(-1)

    # --- harmonics (smooth order) ---
    coef = _pinv_for(T) @ x                                  # (5, 13)
    harmonic = coef.T.reshape(-1)                            # band-major

    # --- NDVI phenology ---
    red = x[:, RED_IDX]
    nir = x[:, NIR_IDX]
    ndvi = (nir - red) / (nir + red + EPS)

    peak = float(ndvi.max())
    argmax = float(ndvi.argmax()) / max(T - 1, 1)            # normalised timing
    mean_ = float(ndvi.mean())
    integral = float(ndvi.sum()) / T
    lo = float(ndvi.min())
    amplitude = peak - lo

    k = int(ndvi.argmax())
    greenup = (ndvi[k] - ndvi[0]) / max(k, 1)
    senescence = (ndvi[-1] - ndvi[k]) / max(T - 1 - k, 1)

    ndvi_feats = np.array([peak, argmax, mean_, integral,
                           float(greenup), float(senescence), amplitude])

    return np.concatenate([basic, segment, harmonic, ndvi_feats])


def sanity_check_ndvi(dataset, n=500, seed=0):
    """
    Falsify the band-order assumption.

    Crops in a growing season must show clearly positive peak NDVI (typically
    0.4-0.9) with the peak somewhere in the middle of the series, not at the
    very start or end. Near-zero, negative, or edge-peaked values mean
    RED_IDX / NIR_IDX are wrong.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)

    peaks, timings = [], []
    for i in idx:
        x = np.asarray(dataset[int(i)][0], dtype=np.float64)
        red, nir = x[:, RED_IDX], x[:, NIR_IDX]
        ndvi = (nir - red) / (nir + red + EPS)
        peaks.append(ndvi.max())
        timings.append(ndvi.argmax() / max(len(ndvi) - 1, 1))

    peaks = np.array(peaks)
    timings = np.array(timings)

    print(f"NDVI sanity check on {len(idx)} parcels "
          f"(assuming red=idx{RED_IDX}, nir=idx{NIR_IDX})")
    print(f"  peak NDVI   mean={peaks.mean():.3f}  "
          f"median={np.median(peaks):.3f}  "
          f"5th={np.percentile(peaks, 5):.3f}  95th={np.percentile(peaks, 95):.3f}")
    print(f"  peak timing median={np.median(timings):.3f} "
          f"(0=season start, 1=season end)")

    ok = peaks.mean() > 0.2 and 0.1 < np.median(timings) < 0.9
    print(f"  => {'PLAUSIBLE' if ok else 'IMPLAUSIBLE - band indices likely wrong'}")
    return ok