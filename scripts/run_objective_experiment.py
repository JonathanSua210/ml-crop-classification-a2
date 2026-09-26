"""
Compare the built-in softmax objective against the custom focal loss,
under both the random split and the held-out-region split.

This is the criterion-B experiment: does a loss function designed around
the observed failure mode (rare classes swamped by majority classes)
actually fix that failure mode?
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.focal import make_xgb_objective

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SAMPLE_FRAC = 1.0        # 0.15 for fast iteration, 1.0 for reported numbers
RARE_CLASSES = [4, 6]    # sunflower, nuts - too few samples to evaluate
HOLDOUT_REGION = "frh04"
NUM_ROUNDS = 200
GAMMA = 2.0
SEED = 42

CLASS_NAMES = ["barley", "wheat", "rapeseed", "corn",
               "orchards", "permanent meadows", "temporary meadows"]

# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
df = pd.read_parquet("data/all_regions_features.parquet")

if SAMPLE_FRAC < 1.0:
    df, _ = train_test_split(
        df, train_size=SAMPLE_FRAC, stratify=df["label"], random_state=SEED
    )
    df = df.reset_index(drop=True)
    print(f"DEV MODE: subsampled to {SAMPLE_FRAC:.0%} -> {df.shape}")

df = df[~df["label"].isin(RARE_CLASSES)].reset_index(drop=True)
label_map = {old: new for new, old in enumerate(sorted(df["label"].unique()))}
df["label"] = df["label"].map(label_map)

feature_cols = [c for c in df.columns if c not in ("label", "field_id", "region")]
X = df[feature_cols].values
y = df["label"].values
regions = df["region"].values
K = len(np.unique(y))

print(f"{len(df)} parcels, {K} classes, {len(feature_cols)} features")

# Inverse-frequency class weights, normalised to mean 1 so the overall
# gradient scale stays comparable to the unweighted run.
counts = np.bincount(y, minlength=K).astype(np.float64)
alpha = counts.sum() / (K * counts)
alpha = alpha / alpha.mean()
print("class weights (alpha):")
for name, c, a in zip(CLASS_NAMES, counts, alpha):
    print(f"  {name:20s} n={int(c):7d}  alpha={a:6.3f}")


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def run(X_tr, X_te, y_tr, y_te, use_focal, tag):
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dtest = xgb.DMatrix(X_te, label=y_te)

    params = {
        "max_depth": 6,
        "eta": 0.1,
        "num_class": K,
        "nthread": -1,
        "seed": SEED,
    }

    if use_focal:
        # Custom objective: XGBoost optimises whatever grad/hess we hand it.
        # disable_default_eval_metric because mlogloss is not the loss here.
        params["disable_default_eval_metric"] = 1
        obj = make_xgb_objective(gamma=GAMMA, alpha=alpha, num_class=K)
        booster = xgb.train(params, dtrain, num_boost_round=NUM_ROUNDS, obj=obj)
    else:
        params["objective"] = "multi:softmax"
        params["eval_metric"] = "mlogloss"
        booster = xgb.train(params, dtrain, num_boost_round=NUM_ROUNDS)

    raw = booster.predict(dtest, output_margin=True)
    if raw.ndim == 1:
        raw = raw.reshape(len(y_te), K)
    preds = raw.argmax(axis=1)

    acc = accuracy_score(y_te, preds)
    mf1 = f1_score(y_te, preds, average="macro")
    print(f"\n=== {tag} ===")
    print(f"accuracy: {acc:.4f}   macro F1: {mf1:.4f}")
    return acc, mf1, preds, y_te


# Splits
Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
tr_mask = regions != HOLDOUT_REGION
te_mask = regions == HOLDOUT_REGION

results = {}
results["random / softmax"] = run(Xtr_r, Xte_r, ytr_r, yte_r, False,
                                  "Random split, built-in softmax")
results["random / focal"] = run(Xtr_r, Xte_r, ytr_r, yte_r, True,
                                "Random split, custom focal")
results["region / softmax"] = run(X[tr_mask], X[te_mask], y[tr_mask], y[te_mask],
                                  False, "Held-out region, built-in softmax")
results["region / focal"] = run(X[tr_mask], X[te_mask], y[tr_mask], y[te_mask],
                                True, "Held-out region, custom focal")

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
print("\n" + "=" * 52)
print(f"{'configuration':28s} {'accuracy':>10s} {'macro F1':>10s}")
print("-" * 52)
for k, (acc, mf1, _, _) in results.items():
    print(f"{k:28s} {acc:10.4f} {mf1:10.4f}")

print("\n=== per-class, held-out region, custom focal ===")
_, _, p_rf, y_rf = results["region / focal"]
print(classification_report(y_rf, p_rf, target_names=CLASS_NAMES,
                            digits=3, zero_division=0))

print("=== per-class, held-out region, built-in softmax (for contrast) ===")
_, _, p_rs, y_rs = results["region / softmax"]
print(classification_report(y_rs, p_rs, target_names=CLASS_NAMES,
                            digits=3, zero_division=0))