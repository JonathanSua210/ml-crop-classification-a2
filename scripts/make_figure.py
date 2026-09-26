"""
The central figure.

Left panel:  regional accuracy per fold, by feature set.
             Shows that temporal features buy ~9 points.
Right panel: random-minus-regional gap per fold, by feature set.
             Shows the fan-out: richer features barely move the MEAN gap
             but spread the folds apart, and frh04 drops to zero.

Reads data/featureset_perfold.csv, so rerunning the experiment at full
size and then rerunning this script redraws with the reported numbers.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC = Path("data/featureset_perfold.csv")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

ORDER = ["basic (order-free)", "basic + ndvi phenology",
         "basic + harmonic", "basic + segment", "all (full temporal)"]
SHORT = {"basic (order-free)": "basic\n(52)",
         "basic + ndvi phenology": "+ NDVI\n(59)",
         "basic + harmonic": "+ harmonic\n(117)",
         "basic + segment": "+ segment\n(130)",
         "all (full temporal)": "all\n(202)"}
REGIONS = ["frh01", "frh02", "frh03", "frh04"]
COLORS = {"frh01": "#4C72B0", "frh02": "#DD8452",
          "frh03": "#55A868", "frh04": "#C44E52"}

df = pd.read_csv(SRC)
df = df[df.feature_set.isin(ORDER)]
xs = np.arange(len(ORDER))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))


def per_region(metric):
    piv = df.pivot(index="feature_set", columns="fold", values=metric)
    return piv.reindex(ORDER)[REGIONS]


# ---------------------------------------------------------------- left
acc = per_region("regional_acc")
for r in REGIONS:
    ax1.plot(xs, acc[r].values, marker="o", color=COLORS[r],
             lw=1.5, alpha=0.85, label=r)
ax1.plot(xs, acc.mean(axis=1).values, color="black", lw=2.5,
         marker="s", label="mean")
ax1.set_xticks(xs)
ax1.set_xticklabels([SHORT[s] for s in ORDER])
ax1.set_ylabel("regional accuracy (held-out region)")
gain = (acc.mean(axis=1).max() - acc.mean(axis=1).iloc[0]) * 100
ax1.set_title(f"Temporal features: {gain:+.1f} points of accuracy")
ax1.grid(axis="y", alpha=0.3)
ax1.legend(fontsize=8, loc="lower right")

# ---------------------------------------------------------------- right
gap = per_region("gap_acc")
for r in REGIONS:
    ax2.plot(xs, gap[r].values, marker="o", color=COLORS[r],
             lw=1.5, alpha=0.85, label=r)
mean_gap = gap.mean(axis=1).values
std_gap = gap.std(axis=1, ddof=1).values
ax2.plot(xs, mean_gap, color="black", lw=2.5, marker="s", label="mean")
ax2.fill_between(xs, mean_gap - std_gap, mean_gap + std_gap,
                 color="grey", alpha=0.18, label="mean ± 1 sd")
ax2.axhline(0, color="black", lw=0.8, ls=":")
ax2.set_xticks(xs)
ax2.set_xticklabels([SHORT[s] for s in ORDER])
ax2.set_ylabel("leakage gap  (random acc − regional acc)")
ratio = std_gap.max() / std_gap[0]
ax2.set_title(f"Mean gap barely moves; fold spread grows {ratio:.1f}x")
ax2.grid(axis="y", alpha=0.3)
ax2.legend(fontsize=8, loc="upper left")

# annotate the spread directly so the claim does not rely on reading the band
for x, s in zip(xs, std_gap):
    ax2.annotate(f"sd {s:.3f}", (x, ax2.get_ylim()[0]),
                 xytext=(0, 4), textcoords="offset points",
                 ha="center", fontsize=7, color="dimgrey")

fig.suptitle("BreizhCrops, leave-one-region-out, XGBoost",
             fontsize=10, color="dimgrey", y=0.995)
fig.tight_layout()

png = OUT_DIR / "featureset_leakage.png"
fig.savefig(png, dpi=200, bbox_inches="tight")
print(f"wrote {png}")