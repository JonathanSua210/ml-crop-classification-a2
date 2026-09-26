"""
Does the leakage gap depend on how much temporal structure the model can see?

Entry 5 hypothesis: the modest gap under order-free features is partly an
artefact of those features destroying phenology timing - the locally varying
signal a model would overfit to. A model that cannot see local structure
cannot leak on it.

STATISTICS NOTE
---------------
An earlier version compared each feature set's mean gap against the baseline
using the pooled fold spread. That was the wrong test: the same four regions
appear in every feature set, so the observations are PAIRED. An unpaired
comparison discards the pairing and is badly underpowered at n=4 - it
reported "inconclusive" for an effect a paired t-test detects at p=0.025.
This version keeps per-fold results and runs a paired test.

With four folds the test has 3 degrees of freedom, so p-values are
indicative rather than strong evidence. The per-fold sign pattern is
reported alongside, since 4/4 folds agreeing is arguably more persuasive
here than the p-value.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.features_v2 import FAMILIES

# ----------------------------------------------------------------------
SAMPLE_FRAC = 1.0      # 1.0 for reported numbers
RARE_CLASSES = [4, 6]
NUM_ROUNDS = 200
MAX_DEPTH = 6
ETA = 0.1
SEED = 42
REGIONS = ["frh01", "frh02", "frh03", "frh04"]

FEATURE_SETS = {
    "basic (order-free)":     ["basic"],
    "basic + segment":        ["basic", "segment"],
    "basic + harmonic":       ["basic", "harmonic"],
    "basic + ndvi phenology": ["basic", "ndvi"],
    "all (full temporal)":    ["basic", "segment", "harmonic", "ndvi"],
}
BASELINE_SET = "basic (order-free)"

# ----------------------------------------------------------------------
df = pd.read_parquet("data/all_regions_features_v2.parquet")

if SAMPLE_FRAC < 1.0:
    df, _ = train_test_split(df, train_size=SAMPLE_FRAC,
                             stratify=df["label"], random_state=SEED)
    df = df.reset_index(drop=True)
    print(f"DEV MODE: {SAMPLE_FRAC:.0%} -> {len(df)} parcels\n")

df = df[~df["label"].isin(RARE_CLASSES)].reset_index(drop=True)
label_map = {o: n for n, o in enumerate(sorted(df["label"].unique()))}
df["label"] = df["label"].map(label_map)

y = df["label"].values
regions = df["region"].values
K = len(np.unique(y))


def fit_predict(X_tr, y_tr, X_te):
    params = {"max_depth": MAX_DEPTH, "eta": ETA, "num_class": K,
              "objective": "multi:softmax", "eval_metric": "mlogloss",
              "nthread": -1, "seed": SEED}
    bst = xgb.train(params, xgb.DMatrix(X_tr, label=y_tr), NUM_ROUNDS)
    raw = bst.predict(xgb.DMatrix(X_te), output_margin=True)
    if raw.ndim == 1:
        raw = raw.reshape(X_te.shape[0], K)
    return raw.argmax(axis=1)


records = []

for set_name, families in FEATURE_SETS.items():
    cols = [c for fam in families for c in FAMILIES[fam]]
    X = df[cols].values
    print(f"=== {set_name}  ({len(cols)} features) ===")

    for held_out in REGIONS:
        te = regions == held_out
        tr = ~te
        n_tr, n_te = int(tr.sum()), int(te.sum())

        p = fit_predict(X[tr], y[tr], X[te])
        ra = accuracy_score(y[te], p)
        rf = f1_score(y[te], p, average="macro")

        i_tr, i_te = train_test_split(np.arange(len(y)), train_size=n_tr,
                                      test_size=n_te, random_state=SEED,
                                      stratify=y)
        pc = fit_predict(X[i_tr], y[i_tr], X[i_te])
        ca = accuracy_score(y[i_te], pc)
        cf = f1_score(y[i_te], pc, average="macro")

        records.append(dict(feature_set=set_name, n_features=len(cols),
                            fold=held_out,
                            regional_acc=ra, random_acc=ca, gap_acc=ca - ra,
                            regional_f1=rf, random_f1=cf, gap_f1=cf - rf))
        print(f"  {held_out}: regional={ra:.4f}  random={ca:.4f}  "
              f"gap={ca - ra:+.4f}")
    print()

per_fold = pd.DataFrame(records)
per_fold.to_csv("data/featureset_perfold.csv", index=False)

# ----------------------------------------------------------------------
agg = (per_fold.groupby("feature_set", sort=False)
       .agg(n=("n_features", "first"),
            regional_acc=("regional_acc", "mean"),
            random_acc=("random_acc", "mean"),
            gap_acc=("gap_acc", "mean"),
            gap_acc_std=("gap_acc", lambda s: s.std(ddof=1)),
            gap_f1=("gap_f1", "mean"),
            gap_f1_std=("gap_f1", lambda s: s.std(ddof=1)))
       .reset_index())
agg.to_csv("data/featureset_summary.csv", index=False)

print("=" * 90)
print(f"{'feature set':26s}{'n':>5s}{'regional':>11s}{'random':>10s}"
      f"{'GAP acc':>14s}{'gap std':>16s}")
print("-" * 90)
for _, r in agg.iterrows():
    print(f"{r.feature_set:26s}{int(r.n):5d}{r.regional_acc:11.4f}"
          f"{r.random_acc:10.4f}{r.gap_acc:+14.4f}{r.gap_acc_std:16.4f}")

# ----------------------------------------------------------------------
print("\n" + "=" * 90)
print("PAIRED COMPARISON vs baseline feature set (n=4 folds, 3 df)")
print("=" * 90)

base = per_fold[per_fold.feature_set == BASELINE_SET].set_index("fold")

print(f"{'feature set':26s}{'d(gap)':>10s}{'t':>8s}{'p':>9s}{'signs':>8s}   per-fold d")
for set_name in FEATURE_SETS:
    if set_name == BASELINE_SET:
        continue
    cur = per_fold[per_fold.feature_set == set_name].set_index("fold")
    a = cur.loc[REGIONS, "gap_acc"].values
    b = base.loc[REGIONS, "gap_acc"].values
    d = a - b
    t, p = stats.ttest_rel(a, b)
    signs = f"{int((d > 0).sum())}/4"
    flag = " *" if p < 0.05 else ""
    print(f"{set_name:26s}{d.mean():+10.4f}{t:8.2f}{p:9.4f}{signs:>8s}   "
          + " ".join(f"{v:+.4f}" for v in d) + flag)

print("\n* = paired t-test p < 0.05. With 3 df treat as indicative;")
print("  the sign pattern (how many of 4 folds agree) is the more robust signal.")

# ----------------------------------------------------------------------
print("\n" + "=" * 90)
print("VARIANCE OF TRANSFER")
print("=" * 90)
print("Does richer featurisation make cross-region transfer less PREDICTABLE,")
print("even where it does not make it worse on average?\n")
b_std = float(agg.loc[agg.feature_set == BASELINE_SET, "gap_acc_std"].iloc[0])
for _, r in agg.iterrows():
    print(f"  {r.feature_set:26s} gap std {r.gap_acc_std:.4f}   "
          f"{r.gap_acc_std / b_std:4.1f}x baseline")

print("\nper-fold gaps by feature set (watch for folds that diverge):")
piv = per_fold.pivot(index="feature_set", columns="fold", values="gap_acc")
piv = piv.reindex(list(FEATURE_SETS.keys()))[REGIONS]
print(piv.to_string(float_format=lambda v: f"{v:+.4f}"))

print("\nwrote data/featureset_perfold.csv and data/featureset_summary.csv")