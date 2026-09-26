"""Build a flat feature table across all four BreizhCrops NUTS-3 regions."""
import pandas as pd
import breizhcrops as bzh
from src.features import extract_features, FEATURE_NAMES

REGIONS = ["frh01", "frh02", "frh03", "frh04"]

all_frames = []

for region in REGIONS:
    print(f"=== {region} ===")
    dataset = bzh.BreizhCrops(region)
    print(f"loaded {len(dataset)} parcels")

    rows = []
    labels = []
    field_ids = []
    for i in range(len(dataset)):
        x, y, field_id = dataset[i]
        feats = extract_features(x.numpy())
        rows.append(feats)
        labels.append(int(y))
        field_ids.append(int(field_id))
        if i % 40000 == 0:
            print(f"  {i}/{len(dataset)}")

    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    X["label"] = labels
    X["field_id"] = field_ids
    X["region"] = region
    all_frames.append(X)

combined = pd.concat(all_frames, ignore_index=True)
combined.to_parquet("data/all_regions_features.parquet")
print(combined.shape)
print(combined["region"].value_counts())
print(combined["label"].value_counts())
