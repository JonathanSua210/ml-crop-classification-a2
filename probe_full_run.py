"""
Predict the cost of a full-mode notebook run, in about 5 minutes.

Measures, on this machine:
  1. feature extraction speed  -> extrapolated to all parcels
  2. one XGBoost fit at full scale (softmax and custom objective)
  3. memory headroom vs the peak the full run needs

Run from the project root:   python probe_full_run.py
"""
import sys, time, gc
import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

import xgboost as xgb

N_FEATURES = 202
N_CLASSES = 7
PROBE_PARCELS = 1500        # for the extraction timing
PROBE_ROUNDS = 10           # scaled up to 200 afterwards
ROUNDS_FULL = 200


def gb(x):
    return x / 1024**3


def mem():
    if psutil is None:
        return None, None
    v = psutil.virtual_memory()
    return gb(v.available), gb(v.total)


print("=" * 66)
print("PROBE: predicting the cost of a full-mode run")
print("=" * 66)

avail, total = mem()
if avail is not None:
    print(f"memory: {avail:.1f} GB available of {total:.1f} GB total")
    if avail < 4:
        print("  WARNING: under 4 GB free. Close other applications before the full run.")
else:
    print("memory: psutil not installed, skipping memory check")

# ---------------------------------------------------------------- 1. extraction
print("\n[1/3] feature extraction speed")
try:
    sys.path.insert(0, ".")
    import breizhcrops as bzh
    from src.features_v2 import extract_features_v2 as extract
except Exception as e:
    print(f"  could not import ({e}); using the notebook's copy is fine, "
          f"this probe just needs src/features_v2.py")
    raise SystemExit(1)

t0 = time.time()
ds = bzh.BreizhCrops("frh04")
load_s = time.time() - t0
n_frh04 = len(ds)
print(f"  loaded frh04 index in {load_s:.1f}s ({n_frh04} parcels)")

t0 = time.time()
for i in range(PROBE_PARCELS):
    extract(np.asarray(ds[i][0], dtype=np.float64))
per_parcel = (time.time() - t0) / PROBE_PARCELS
print(f"  {per_parcel*1e3:.2f} ms per parcel")

TOTAL_PARCELS = 608263
extract_min = per_parcel * TOTAL_PARCELS / 60
print(f"  -> all {TOTAL_PARCELS:,} parcels: {extract_min:.0f} min")

# ---------------------------------------------------------------- 2. fit speed
print(f"\n[2/3] XGBoost fit speed at full scale ({PROBE_ROUNDS} rounds, scaled to {ROUNDS_FULL})")
N_TRAIN = 456_000        # a typical LORO training set (3 of 4 regions)
rng = np.random.default_rng(0)
X = rng.random((N_TRAIN, N_FEATURES), dtype=np.float32)
yy = rng.integers(0, N_CLASSES, N_TRAIN)
d = xgb.DMatrix(X, label=yy)

base = {"max_depth": 6, "eta": 0.1, "num_class": N_CLASSES, "nthread": -1, "seed": 0}

t0 = time.time()
xgb.train({**base, "objective": "multi:softmax", "eval_metric": "mlogloss"}, d, PROBE_ROUNDS)
soft_full = (time.time() - t0) / PROBE_ROUNDS * ROUNDS_FULL
print(f"  softmax, 202 features: {soft_full/60:.1f} min per fit")


def grad_hess(preds, dtrain):
    y = dtrain.get_label().astype(np.int64)
    z = np.asarray(preds, dtype=np.float64).reshape(y.size, N_CLASSES)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    oh = np.zeros_like(p)
    oh[np.arange(y.size), y] = 1.0
    return p - oh, np.maximum(p * (1 - p), 1e-6)


t0 = time.time()
xgb.train({**base, "disable_default_eval_metric": 1}, d, PROBE_ROUNDS, obj=grad_hess)
custom_full = (time.time() - t0) / PROBE_ROUNDS * ROUNDS_FULL
print(f"  custom objective:      {custom_full/60:.1f} min per fit")

peak_probe = psutil.Process().memory_info().rss if psutil else None
del X, d
gc.collect()

# ---------------------------------------------------------------- 3. totals
print("\n[3/3] projected full run")

# Experiment 1: 4 folds x 2 objectives x 2 splits, 52 features (~4x faster than 202)
exp1 = 4 * 2 * 2 * ((soft_full + custom_full) / 2) / 4
# Experiment 2: 5 feature sets x 4 folds x 2 splits, softmax, average ~120 features
exp2 = 5 * 4 * 2 * soft_full * 0.6
# Experiment 3: 4 folds x (3 inner x 4 gammas at 30% data + 1 final custom + 1 final softmax)
exp3 = 4 * (3 * 4 * custom_full * 0.30 + custom_full + soft_full)

total_min = extract_min + (exp1 + exp2 + exp3) / 60
print(f"  feature extraction   {extract_min:6.0f} min")
print(f"  experiment 1         {exp1/60:6.0f} min")
print(f"  experiment 2         {exp2/60:6.0f} min")
print(f"  experiment 3         {exp3/60:6.0f} min")
print(f"  {'-'*34}")
print(f"  TOTAL                {total_min:6.0f} min   ({total_min/60:.1f} hours)")

peak_gb = TOTAL_PARCELS * N_FEATURES * 4 / 1024**3
print(f"\n  feature table in memory: {peak_gb:.1f} GB")
print(f"  expected peak usage:     {peak_gb*3:.1f} GB (table + a working copy + DMatrix)")
if avail is not None:
    verdict = "should fit" if avail > peak_gb * 3 + 1 else "TIGHT - close other apps first"
    print(f"  available now:           {avail:.1f} GB  -> {verdict}")

print("\nNotes:")
print("  - extraction is cached to data/nb_features_full.parquet, so a crash")
print("    after that point does not repeat it on the next run")
print("  - nbconvert only writes the notebook at the end; if it fails the")
print("    original file is left untouched")