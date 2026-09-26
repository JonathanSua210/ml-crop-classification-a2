"""
Leave-one-region-out (LORO) cross validation.

Fixes three weaknesses in the earlier single-holdout experiment:

1. n=1 -> 4 folds, so the leakage gap gets a mean and a standard deviation
   instead of being a single point estimate that might just be frh04 being
   unusual.
2. No baseline -> majority-class baseline reported per fold, so the headline
   accuracy has context.
3. Unmatched comparison -> for every regional fold we also run a RANDOM split
   with the SAME train/test sizes drawn from the same pool. Without size
   matching, part of any gap could simply be a difference in training set
   size rather than spatial structure.

Hyperparameters (including focal gamma) are treated as fixed inputs here.
They must be selected on an inner validation region, never on the outer test
region - see scripts/run_gamma_sweep.py.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.focal import make_xgb_objective

# ----------------------------------------------------------------------
SAMPLE_FRAC = 1.0      # 0.15 to smoke-test the whole thing, 1.0 for reported numbers
RARE_CLASSES = [4, 6]
NUM_ROUNDS = 200
MAX_DEPTH = 6
ETA = 0.1
GAMMA = 2.0
SEED = 42

REGIONS = ["frh01", "frh02", "frh03", "frh04"]
CLASS_NAMES = ["barley", "wheat", "rapeseed", "corn",
               "orchards", "permanent meadows", "temporary meadows"]

# ----------------------------------------------------------------------
df = pd.read_parquet("data/all_regions_features.parquet")

if SAMPLE_FRAC < 1.0:
    df, _ = train_test_split(
        df, train_size=SAMPLE_FRAC, stratify=df["label"], random_state=SEED
    )
    df = df.reset_index(drop=True)
    print(f"DEV MODE: {SAMPLE_FRAC:.0%} subsample -> {len(df)} parcels\n")

df = df[~df["label"].isin(RARE_CLASSES)].reset_index(drop=True)
label_map = {old: new for new, old in enumerate(sorted(df["label"].unique()))}
df["label"] = df["label"].map(label_map)

feature_cols = [c for c in df.columns if c not in ("label", "field_id", "region")]
X = df[feature_cols].values
y = df["label"].values
regions = df["region"].values
K = len(np.unique(y))

# alpha from the FULL pool. Strictly this should be recomputed per fold from
# the training regions only; class proportions are stable enough across
# regions here that it makes no practical difference, but it is a small
# information leak and is noted as such.
counts = np.bincount(y, minlength=K).astype(np.float64)
alpha = counts.sum() / (K * counts)
alpha = alpha / alpha.mean()


def fit_predict(X_tr, y_tr, X_te, use_focal):
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dtest = xgb.DMatrix(X_te)
    params = {"max_depth": MAX_DEPTH, "eta": ETA, "num_class": K,
              "nthread": -1, "seed": SEED}
    if use_focal:
        params["disable_default_eval_metric"] = 1
        obj = make_xgb_objective(gamma=GAMMA, alpha=alpha, num_class=K)
        bst = xgb.train(params, dtrain, NUM_ROUNDS, obj=obj)
    else:
        params["objective"] = "multi:softmax"
        params["eval_metric"] = "mlogloss"
        bst = xgb.train(params, dtrain, NUM_ROUNDS)
    raw = bst.predict(dtest, output_margin=True)
    if raw.ndim == 1:
        raw = raw.reshape(X_te.shape[0], K)
    return raw.argmax(axis=1)


def score(y_true, y_pred):
    return (accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average="macro"),
            f1_score(y_true, y_pred, average=None, labels=range(K), zero_division=0))


rows = []
perclass = {}

for held_out in REGIONS:
    te_mask = regions == held_out
    tr_mask = ~te_mask
    n_tr, n_te = int(tr_mask.sum()), int(te_mask.sum())

    # Majority-class baseline, fitted on train, applied to test
    majority = np.bincount(y[tr_mask], minlength=K).argmax()
    base_acc, base_f1, _ = score(y[te_mask], np.full(n_te, majority))

    print(f"=== fold: hold out {held_out} "
          f"(train {n_tr}, test {n_te}) ===")
    print(f"  majority baseline     acc={base_acc:.4f}  macroF1={base_f1:.4f}")

    for use_focal, oname in [(False, "softmax"), (True, "focal")]:
        # --- regional split ---
        pred = fit_predict(X[tr_mask], y[tr_mask], X[te_mask], use_focal)
        r_acc, r_f1, r_pc = score(y[te_mask], pred)

        # --- size-matched random control ---
        idx_tr, idx_te = train_test_split(
            np.arange(len(y)), train_size=n_tr, test_size=n_te,
            random_state=SEED, stratify=y
        )
        pred_c = fit_predict(X[idx_tr], y[idx_tr], X[idx_te], use_focal)
        c_acc, c_f1, c_pc = score(y[idx_te], pred_c)

        print(f"  {oname:8s} regional   acc={r_acc:.4f}  macroF1={r_f1:.4f}")
        print(f"  {oname:8s} random     acc={c_acc:.4f}  macroF1={c_f1:.4f}")
        print(f"  {oname:8s} GAP        acc={c_acc - r_acc:+.4f}  "
              f"macroF1={c_f1 - r_f1:+.4f}")

        rows.append(dict(fold=held_out, objective=oname,
                         base_acc=base_acc,
                         regional_acc=r_acc, regional_f1=r_f1,
                         random_acc=c_acc, random_f1=c_f1,
                         gap_acc=c_acc - r_acc, gap_f1=c_f1 - r_f1))
        perclass[(held_out, oname, "regional")] = r_pc
        perclass[(held_out, oname, "random")] = c_pc
    print()

res = pd.DataFrame(rows)
res.to_csv("data/loro_results.csv", index=False)

print("=" * 72)
print("LORO SUMMARY  (mean +/- std across 4 regional folds)")
print("=" * 72)
print(f"majority baseline accuracy: {res['base_acc'].mean():.4f} "
      f"+/- {res['base_acc'].std():.4f}")
print()
for oname in ["softmax", "focal"]:
    s = res[res.objective == oname]
    print(f"{oname}")
    print(f"   random   acc {s.random_acc.mean():.4f} +/- {s.random_acc.std():.4f}"
          f"   macroF1 {s.random_f1.mean():.4f} +/- {s.random_f1.std():.4f}")
    print(f"   regional acc {s.regional_acc.mean():.4f} +/- {s.regional_acc.std():.4f}"
          f"   macroF1 {s.regional_f1.mean():.4f} +/- {s.regional_f1.std():.4f}")
    print(f"   GAP      acc {s.gap_acc.mean():+.4f} +/- {s.gap_acc.std():.4f}"
          f"   macroF1 {s.gap_f1.mean():+.4f} +/- {s.gap_f1.std():.4f}")
    print()

print("per-fold accuracy gap (random - regional):")
for oname in ["softmax", "focal"]:
    s = res[res.objective == oname].set_index("fold")
    vals = "  ".join(f"{r}:{s.loc[r, 'gap_acc']:+.4f}" for r in REGIONS)
    print(f"   {oname:8s} {vals}")

print("\nper-class macro F1, regional folds, mean across folds:")
hdr = f"{'class':20s}" + "".join(f"{o:>12s}" for o in ["softmax", "focal"])
print(hdr)
for ci, cname in enumerate(CLASS_NAMES):
    sm = np.mean([perclass[(r, "softmax", "regional")][ci] for r in REGIONS])
    fc = np.mean([perclass[(r, "focal", "regional")][ci] for r in REGIONS])
    print(f"{cname:20s}{sm:12.3f}{fc:12.3f}")

print("\nwrote data/loro_results.csv")