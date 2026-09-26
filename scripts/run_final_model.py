"""
The final model, evaluated with nested leave-one-region-out.

Combines the two improvements that were previously only tested separately:
  - all 202 temporal features   (+9 accuracy points, Entry 10)
  - focal loss                  (+~4 macro F1 points, Entry 7)

and closes three methodological gaps:

1. GAMMA WAS ASSERTED, NOT SELECTED.  gamma is now chosen per outer fold by
   an INNER leave-one-region-out over the three training regions. The outer
   test region is never seen during selection.

   Why inner LORO rather than one validation region: the project's own main
   finding (Entries 9-12) is that single-region validation is unreliable for
   exactly this kind of model - fold-to-fold spread grows 4-6x under focal
   loss and temporal features. Selecting gamma on one validation region would
   mean selecting on noise. So the validation design follows directly from
   the result it is validating.

2. ALPHA LEAKED.  Class weights were computed from all four regions,
   including the test region (flagged in Entry 9). They are now computed per
   outer fold from the training regions only, and per inner fold from the
   inner-training regions only.

3. NO CONFUSION MATRIX.  Predictions are pooled across the four outer folds
   (each parcel is predicted exactly once, by a model that never saw its
   region) and a row-normalised confusion matrix is saved.

Selection runs on a stratified subsample of each inner training set
(SELECT_FRAC) to keep the 48 inner fits affordable; final models are fitted
on the full training regions. This is a deliberate cost trade-off and is
documented as such.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from src.features_v2 import FEATURE_NAMES_V2
from src.focal import make_xgb_objective

# ----------------------------------------------------------------------
GAMMA_GRID = [0.0, 1.0, 2.0, 3.0]   # 0.0 = class-weighted cross entropy
SELECT_FRAC = 0.30                   # subsample used for inner selection only
SELECT_METRIC = "macro_f1"           # see note at the bottom of the output
RARE_CLASSES = [4, 6]
NUM_ROUNDS = 200
MAX_DEPTH = 6
ETA = 0.1
SEED = 42
REGIONS = ["frh01", "frh02", "frh03", "frh04"]
CLASS_NAMES = ["barley", "wheat", "rapeseed", "corn",
               "orchards", "permanent meadows", "temporary meadows"]

Path("figures").mkdir(exist_ok=True)

# ----------------------------------------------------------------------
df = pd.read_parquet("data/all_regions_features_v2.parquet")
df = df[~df["label"].isin(RARE_CLASSES)].reset_index(drop=True)
label_map = {o: n for n, o in enumerate(sorted(df["label"].unique()))}
df["label"] = df["label"].map(label_map)

X = df[FEATURE_NAMES_V2].values
y = df["label"].values
regions = df["region"].values
K = len(label_map)
print(f"{len(y)} parcels, {K} classes, {X.shape[1]} features\n")


def class_weights(labels):
    """Inverse-frequency weights, mean-normalised. Computed from TRAINING labels only."""
    c = np.bincount(labels, minlength=K).astype(np.float64)
    c = np.maximum(c, 1.0)                     # guard a class absent from a subset
    a = c.sum() / (K * c)
    return a / a.mean()


def fit_predict(X_tr, y_tr, X_te, gamma=None, alpha=None):
    """gamma=None -> built-in softmax. Otherwise focal with the given gamma/alpha."""
    params = {"max_depth": MAX_DEPTH, "eta": ETA, "num_class": K,
              "nthread": -1, "seed": SEED}
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    if gamma is None:
        params.update(objective="multi:softmax", eval_metric="mlogloss")
        bst = xgb.train(params, dtrain, NUM_ROUNDS)
    else:
        params["disable_default_eval_metric"] = 1
        obj = make_xgb_objective(gamma=gamma, alpha=alpha, num_class=K)
        bst = xgb.train(params, dtrain, NUM_ROUNDS, obj=obj)
    raw = bst.predict(xgb.DMatrix(X_te), output_margin=True)
    if raw.ndim == 1:
        raw = raw.reshape(X_te.shape[0], K)
    return raw.argmax(axis=1)


def metric(y_true, y_pred):
    if SELECT_METRIC == "macro_f1":
        return f1_score(y_true, y_pred, average="macro")
    return accuracy_score(y_true, y_pred)


def subsample(idx, frac):
    if frac >= 1.0:
        return idx
    keep, _ = train_test_split(idx, train_size=frac, stratify=y[idx],
                               random_state=SEED)
    return keep


# ----------------------------------------------------------------------
fold_rows = []
selection_rows = []
pooled_true, pooled_focal, pooled_soft, pooled_region = [], [], [], []

for outer in REGIONS:
    te = np.where(regions == outer)[0]
    tr = np.where(regions != outer)[0]
    inner_regions = [r for r in REGIONS if r != outer]
    print(f"=== outer fold: test on {outer} ===")

    # ---------------- inner LORO: select gamma ----------------
    scores = {g: [] for g in GAMMA_GRID}
    for inner in inner_regions:
        v_idx = tr[regions[tr] == inner]
        t_idx = subsample(tr[regions[tr] != inner], SELECT_FRAC)
        a_inner = class_weights(y[t_idx])
        for g in GAMMA_GRID:
            p = fit_predict(X[t_idx], y[t_idx], X[v_idx], gamma=g, alpha=a_inner)
            s = metric(y[v_idx], p)
            scores[g].append(s)
            selection_rows.append(dict(outer=outer, inner_val=inner,
                                       gamma=g, score=s))

    mean_scores = {g: float(np.mean(v)) for g, v in scores.items()}
    best_g = max(mean_scores, key=mean_scores.get)
    print("  inner selection (" + SELECT_METRIC + ", mean over 3 inner folds):")
    for g in GAMMA_GRID:
        sd = float(np.std(scores[g], ddof=1))
        mark = "  <- selected" if g == best_g else ""
        print(f"    gamma={g:.1f}  {mean_scores[g]:.4f} +/- {sd:.4f}{mark}")

    # ---------------- final fits on full training regions ----------------
    a_outer = class_weights(y[tr])
    p_focal = fit_predict(X[tr], y[tr], X[te], gamma=best_g, alpha=a_outer)
    p_soft = fit_predict(X[tr], y[tr], X[te], gamma=None)

    row = dict(fold=outer, gamma=best_g,
               focal_acc=accuracy_score(y[te], p_focal),
               focal_f1=f1_score(y[te], p_focal, average="macro"),
               soft_acc=accuracy_score(y[te], p_soft),
               soft_f1=f1_score(y[te], p_soft, average="macro"))
    fold_rows.append(row)
    print(f"  final  softmax acc={row['soft_acc']:.4f} F1={row['soft_f1']:.4f}   "
          f"focal(g={best_g:.1f}) acc={row['focal_acc']:.4f} F1={row['focal_f1']:.4f}\n")

    pooled_true.append(y[te])
    pooled_focal.append(p_focal)
    pooled_soft.append(p_soft)
    pooled_region.append(np.full(len(te), outer))

res = pd.DataFrame(fold_rows)
res.to_csv("data/final_model_folds.csv", index=False)
pd.DataFrame(selection_rows).to_csv("data/final_model_selection.csv", index=False)

yt = np.concatenate(pooled_true)
yf = np.concatenate(pooled_focal)
ys = np.concatenate(pooled_soft)
pd.DataFrame(dict(region=np.concatenate(pooled_region), y_true=yt,
                  pred_focal=yf, pred_softmax=ys)
             ).to_csv("data/final_model_predictions.csv", index=False)

# ----------------------------------------------------------------------
print("=" * 72)
print("FINAL MODEL  (all 202 features, nested LORO)")
print("=" * 72)
print(f"gamma selected per fold: "
      + ", ".join(f"{r.fold}={r.gamma:.1f}" for r in res.itertuples()))
for name, a, f in [("softmax", "soft_acc", "soft_f1"),
                   ("focal (selected gamma)", "focal_acc", "focal_f1")]:
    print(f"  {name:24s} acc {res[a].mean():.4f} +/- {res[a].std(ddof=1):.4f}"
          f"   macroF1 {res[f].mean():.4f} +/- {res[f].std(ddof=1):.4f}")

print("\nper-class F1, pooled over all four held-out regions:")
f1s = f1_score(yt, ys, average=None, labels=range(K), zero_division=0)
f1f = f1_score(yt, yf, average=None, labels=range(K), zero_division=0)
print(f"{'class':20s}{'softmax':>10s}{'focal':>10s}{'change':>10s}")
for i, c in enumerate(CLASS_NAMES):
    print(f"{c:20s}{f1s[i]:10.3f}{f1f[i]:10.3f}{f1f[i] - f1s[i]:+10.3f}")


# ----------------------------------------------------------------------
def plot_cm(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=range(K))
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(K):
        for j in range(K):
            v = cmn[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.55 else "black")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return cm


cm_f = plot_cm(yt, yf, "Final model (focal), pooled held-out regions\n"
               "rows normalised: each row sums to 1",
               "figures/confusion_final_focal.png")
cm_s = plot_cm(yt, ys, "Softmax, same features, pooled held-out regions\n"
               "rows normalised: each row sums to 1",
               "figures/confusion_final_softmax.png")

# ----------------------------------------------------------------------
print("\nlargest off-diagonal confusions, focal (share of true class):")
cmn = cm_f / cm_f.sum(axis=1, keepdims=True).clip(min=1)
off = [(cmn[i, j], CLASS_NAMES[i], CLASS_NAMES[j])
       for i in range(K) for j in range(K) if i != j]
for v, t, p in sorted(off, reverse=True)[:6]:
    print(f"  {v:.3f}  true {t:18s} -> predicted {p}")

print("\nNOTE on selection metric: gamma is selected on macro F1, which Entry 8")
print("argues can reward low-precision predictions for rare classes. If the")
print("selected gamma is at the top of the grid, that is worth reading as the")
print("metric pushing toward over-prediction rather than as a clear optimum.")

print("\nwrote data/final_model_folds.csv, data/final_model_selection.csv,")
print("      data/final_model_predictions.csv,")
print("      figures/confusion_final_focal.png, figures/confusion_final_softmax.png")