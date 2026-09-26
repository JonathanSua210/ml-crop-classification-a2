"""
Rebuild the parcel feature table using the v2 (order-aware) features.

Writes data/all_regions_features_v2.parquet alongside the v1 table, so both
featurisations remain available and the comparison can be rerun.

Runs an NDVI band-order sanity check BEFORE the ~15 minute extraction, so a
wrong band index fails in seconds rather than after the full rebuild.
"""
import numpy as np
import pandas as pd
import breizhcrops as bzh

from src.features_v2 import extract_features_v2, FEATURE_NAMES_V2, sanity_check_ndvi

REGIONS = ["frh01", "frh02", "frh03", "frh04"]
OUT = "data/all_regions_features_v2.parquet"

# ------------------------------------------------------------------
# Verify the band-order assumption before doing 15 minutes of work
# ------------------------------------------------------------------
print("verifying band order assumption on frh04...\n")
probe = bzh.BreizhCrops("frh04")
ok = sanity_check_ndvi(probe, n=500)
print()

if not ok:
    raise SystemExit(
        "NDVI values are implausible for vegetated parcels.\n"
        "RED_IDX / NIR_IDX in src/features_v2.py are probably wrong.\n"
        "Inspect the band ordering before rebuilding."
    )

# ------------------------------------------------------------------
# Extraction
# ------------------------------------------------------------------
frames = []

for region in REGIONS:
    print(f"=== {region} ===")
    dataset = bzh.BreizhCrops(region)
    n = len(dataset)
    print(f"loaded {n} parcels")

    feats = np.empty((n, len(FEATURE_NAMES_V2)), dtype=np.float32)
    labels = np.empty(n, dtype=np.int64)
    field_ids = np.empty(n, dtype=np.int64)

    for i in range(n):
        x, y, fid = dataset[i]
        feats[i] = extract_features_v2(x.numpy())
        labels[i] = int(y)
        field_ids[i] = int(fid)
        if i % 40000 == 0:
            print(f"  {i}/{n}")

    df = pd.DataFrame(feats, columns=FEATURE_NAMES_V2)
    df["label"] = labels
    df["field_id"] = field_ids
    df["region"] = region
    frames.append(df)

combined = pd.concat(frames, ignore_index=True)
combined.to_parquet(OUT)

print(f"\nwrote {OUT}")
print(f"shape: {combined.shape}")
print(combined["region"].value_counts())

# quick NaN / inf audit - harmonic fits and NDVI ratios can misbehave
num = combined[FEATURE_NAMES_V2]
n_nan = int(np.isnan(num.values).sum())
n_inf = int(np.isinf(num.values).sum())
print(f"\nNaNs: {n_nan}   infs: {n_inf}")
if n_nan or n_inf:
    bad = num.columns[(np.isnan(num.values) | np.isinf(num.values)).any(axis=0)]
    print(f"affected columns: {list(bad)[:20]}")