# Implementation Log

Crop type classification from Sentinel-2 parcel time series.
Research question: how much of the accuracy reported for parcel-level crop
classification is an artefact of spatially leaky train/test splits?

---

## Entry 1 — Project scoping and dataset selection

**Decision: BreizhCrops over alternatives.**
I needed a dataset that was (a) real and non-trivial, as the spec requires,
and (b) had a genuine spatial grouping structure, because the whole research
question depends on being able to split by region rather than at random.
BreizhCrops fits: ~610k field parcels across Brittany, 9 crop classes, and the
authors partitioned it into four NUTS-3 regions specifically because
spatially adjacent parcels leak between train and test.

I considered using data from work (paddock imagery, cross-paddock splits) but
rejected it: the spec requires a publicly accessible notebook, and that data
is not mine to publish.

**Why machine learning at all.** A rule-based spectral threshold classifier
fails here because the discriminative signal is the *shape of the seasonal
reflectance trajectory*, not any single threshold — wheat and barley have
near-identical spectra at any single date and differ mainly in senescence
timing. That is a learnable pattern and an awkward one to hand-code.

---

## Entry 2 — Environment setup

Windows, PowerShell, Python 3.12.3, venv. Codebase in `src/`, runnable
scripts in `scripts/`.

**Challenge: PowerShell heredoc quoting.**
I was generating Python files from the shell using `@"..."@` here-strings.
This worked until a file contained f-strings with nested single quotes
(`f"{'random split':25s}"`), at which point PowerShell's parser choked on the
escaped-quote soup. Wasted ~15 minutes on escaping before concluding it was
the wrong tool. Switched to writing files in the editor directly. Lesson:
shell heredocs are fine for config, bad for code with nested quotes.

**Challenge: `ModuleNotFoundError: No module named 'src'`.**
Running `python scripts/build_dataset.py` does not put the project root on
`sys.path`, so `from src.features import ...` failed. Fixed by adding
`scripts/__init__.py` and invoking as `python -m scripts.build_dataset`.
I understand why this works: `-m` runs the module with the current directory
prepended to `sys.path`, whereas direct script execution prepends the
*script's* directory instead.

---

## Entry 3 — Featurisation

Each parcel arrives as a `(45, 13)` tensor: 45 Sentinel-2 acquisitions across
the 2017 season, 13 spectral bands each. XGBoost needs a fixed-length vector,
so the time axis has to be collapsed.

**Design decision: per-band mean, std, min, max → 52 features.**
Deliberately the simplest defensible choice, for two reasons. First, it gives
a baseline that later featurisations can be measured against. Second, and more
interestingly, it *discards all temporal ordering* — which turns out to matter
for interpreting the leakage result (see Entry 5).

Known limitation, stated up front: this throws away phenology timing, which is
probably the single most discriminative property for separating cereals.
Percentile features or harmonic fits would retain more. Not yet tested.

**Challenge: Arrow serialisation failure.**
`df.to_parquet()` raised `ArrowInvalid: Could not convert tensor(8)`. The
dataset returns labels as 0-dimensional `torch.Tensor` objects, not Python
ints, and Arrow has no type mapping for those. Fixed with `int(y)`. Also
removed a genuine inefficiency I had introduced: I was re-indexing
`dataset[i]` three separate times per parcel to fetch features, label and
field id. Now captured in a single pass.

**Cost:** full build across all four regions is a one-time ~20 min download
(~5.7 GB compressed) plus feature extraction. Output cached to
`data/all_regions_features.parquet` (608,263 rows × 55 cols) so no subsequent
experiment pays that cost.

---

## Entry 4 — Class imbalance

Label distribution across the full dataset:

| class | count |
|---|---|
| temporary meadows | 182,212 |
| corn | 153,908 |
| permanent meadows | 127,813 |
| wheat | 89,555 |
| barley | 36,905 |
| rapeseed | 14,732 |
| orchards | 3,070 |
| nuts | 49 |
| sunflower | 19 |

Four orders of magnitude between the largest and smallest class.

**Decision: drop sunflower and nuts.** With 19 and 49 examples, a random split
can place nearly all of a class on one side by chance, and any per-class
metric computed on a handful of test samples is noise. The BreizhCrops authors
discarded rare classes on similar grounds, so there is precedent.

**Challenge: XGBoost label contiguity.**
After dropping classes 4 and 6, the remaining labels were `[0,1,2,3,5,7,8]`.
`XGBClassifier` rejected this: it requires labels to be exactly
`0..n_classes-1`. Remapped, keeping an inverse map so results can be reported
against real crop names rather than indices.

**Orchards retained but flagged.** At 3,070 samples it survives, but the model
scores F1 = 0.000 on it — it never predicts the class at all. Kept in
deliberately as an honest negative result rather than quietly dropped.

---

## Entry 5 — Main experiment: split strategy

Identical model (200 trees, depth 6, eta 0.1), two split strategies.

| | accuracy | macro F1 |
|---|---|---|
| random parcel split | 0.5917 | 0.4395 |
| held-out region (frh04) | 0.5733 | 0.4281 |
| **gap** | **0.0184** | **0.0114** |

**Interpretation.** The gap is real and in the predicted direction, but modest
— about 1.8 points of accuracy, not the dramatic collapse the leakage
literature might lead you to expect. Three candidate explanations, which I
think are worth distinguishing rather than picking one:

1. My featurisation is coarse. Collapsing 45 timesteps to four statistics per
   band destroys most of the region-specific temporal detail (local sowing
   dates, weather-driven phenology shifts) that a model *could* have
   overfitted to. A model that never saw that detail cannot leak on it. This
   predicts the gap would *widen* with richer temporal features — a testable
   claim, and the most interesting thread here.
2. Brittany's four NUTS-3 regions are agriculturally homogeneous: one climate
   zone, similar rotations. The domain shift between them is genuinely small.
3. Accuracy is dominated by the three majority classes, which likely behave
   consistently across regions. The macro F1 gap being smaller than the
   accuracy gap is mildly against this, but per-class analysis is the right
   way to check.

**Per-class finding (15% subsample, indicative).** Barley precision drops from
0.510 under random split to 0.365 under regional split, while recall stays
flat (0.097 → 0.114). So under regional evaluation the model issues barley
predictions far more promiscuously. That is a concrete instance of the failure
the research question anticipates. Rapeseed moves the other way (precision
0.747 → 0.842) — not yet explained; possibly a base-rate artefact, since
support differs between the two test sets (442 vs 498).

---

## Entry 6 — Custom objective: multiclass focal loss

**Motivation, grounded in the failure above.** Barley and orchards are being
swamped. Standard softmax cross-entropy weights every sample equally, so the
gradient is dominated by the hundreds of thousands of easy meadow and corn
examples. Focal loss down-weights well-classified samples by a factor
`(1-p_t)^gamma`, concentrating the gradient on hard and rare ones.

XGBoost provides no multiclass focal objective, so it has to be supplied as a
custom `obj` returning gradient and Hessian with respect to the raw margins.

**Theory to code.** With `p = softmax(z)`, `t` the true class, `u = p_t`:

    L = -alpha_t * (1-u)^gamma * log(u)

Differentiating and applying the softmax Jacobian `du/dz_k = u(delta_tk - p_k)`:

    dL/dz_k   = g(u) * u * (delta_tk - p_k)
    d2L/dz_k2 = (g'(u)u + g(u)) * u * (delta_tk - p_k)^2
                - g(u) * u * p_k * (1 - p_k)

where `g(u)` and `g'(u)` are the first and second derivatives of L wrt u,
written out in `src/focal.py`. These map directly onto the `grad` and `hess`
arrays returned to XGBoost.

**Verification — this is the part I would want to be asked about.**
Two independent checks, in `tests/test_focal.py`:

1. *Analytic vs finite difference.* Central differences on the loss surface
   agree with the analytic gradient to ~1e-10 and the analytic Hessian
   diagonal to ~1e-5. The Hessian agreement is looser, but that is expected
   rather than a bug: a second-order central difference has rounding error
   scaling as eps/h^2, which for h=1e-5 puts the noise floor at ~1e-6.
2. *Reduction identity.* At `gamma=0, alpha=1` focal loss is exactly softmax
   cross-entropy, so the derivatives must collapse to `p - onehot` and
   `p(1-p)`. They match **exactly**, to 0.0. This is the stronger check,
   because it compares against a closed form rather than a numerical
   approximation.

**Knowledge gap, stated honestly.** XGBoost consumes only the *diagonal* of
the Hessian; the true Hessian of a softmax loss has off-diagonal terms
(`d2L/dz_j dz_k`, j != k) which are discarded. I derived and verified the
diagonal, but I have not worked through what that truncation costs in terms of
convergence — I know it makes the Newton step approximate rather than exact,
and that XGBoost's built-in multiclass objective makes the identical
approximation, which is why I consider it acceptable here. I have not proved
it is harmless.

**Numerical guard.** I floor the Hessian at 1e-6. Focal loss with gamma > 1
can produce near-zero or slightly negative diagonal curvature for very
confident correct predictions, and XGBoost divides by the Hessian sum when
computing leaf weights, so unguarded values blow up the leaf. The floor is a
standard trick, but it does mean the optimiser is not taking a true Newton
step for those samples.

---

## Entry 7 — Results: custom objective vs built-in

Full dataset, 608,195 parcels, 7 classes, 200 rounds, depth 6, eta 0.1.
Focal gamma = 2.0, alpha = inverse class frequency normalised to mean 1.

| configuration | accuracy | macro F1 |
|---|---|---|
| random / softmax | 0.5917 | 0.4395 |
| random / focal | 0.5745 | 0.4839 |
| region / softmax | 0.5733 | 0.4281 |
| region / focal | 0.5356 | 0.4617 |

**The trade behaved as designed.** Focal loss costs accuracy and buys macro
F1 in both split regimes. That is the intended effect, not a regression:
accuracy is dominated by three majority classes, and the objective was
deliberately reweighted away from them.

**Unexpected and more interesting: the leakage gap roughly doubles.**

| | accuracy gap | macro F1 gap |
|---|---|---|
| softmax | 0.0184 | 0.0114 |
| focal | 0.0389 | 0.0222 |

This resolves the ambiguity left open in Entry 5. One candidate explanation
there was that the modest gap reflected a model that had never learned the
hard classes in the first place, and therefore had no region-specific
structure to fail to transfer. That now looks right. Under softmax, orchards
F1 was exactly 0.000 and barley recall was 0.149 — the model was effectively
declining to predict them. Focal loss forces it to fit those classes, and the
fitted signal transfers poorly across regions.

Stated as a finding: **the harder a model is pushed to attend to rare classes,
the more a random split overstates its deployment performance.** Practitioners
who reweight for imbalance and evaluate on a random split are therefore
compounding two errors, not one.

**Per-class, held-out region (focal vs softmax):**

| class | F1 softmax | F1 focal |
|---|---|---|
| barley | 0.221 | 0.326 |
| wheat | 0.493 | 0.522 |
| rapeseed | 0.550 | 0.642 |
| corn | 0.731 | 0.715 |
| orchards | 0.000 | 0.029 |
| permanent meadows | 0.388 | 0.472 |
| temporary meadows | 0.613 | 0.526 |

Gains concentrate in the under-served classes; the cost is paid almost
entirely by temporary meadows (0.613 → 0.526) and marginally by corn.

**Orchards is an over-correction, not a win.** F1 moved off zero, but
precision is 0.016: about 98% of parcels predicted as orchards are not
orchards. With alpha = 5.05 the model is carpet-bombing the class to recover
67 of 553 true positives. The metric improved; the output got less usable.

---

## Entry 8 — Loss function vs task objective

The orchards result above is the cleanest statement of the discrepancy
between what the model optimises and what the task actually requires.

**What the loss optimises.** Focal loss with inverse-frequency alpha
maximises per-sample classification performance with rare classes upweighted.
Macro F1, the headline evaluation metric, rewards exactly this: it averages
per-class F1 without weighting by class size, so lifting orchards from 0.000
to 0.029 contributes the same amount as lifting corn by 0.029 would.

**What the task requires.** The deployment artefact is a crop map used for
subsidy verification and area statistics. Its value depends on whether a
parcel labelled "orchards" is actually an orchard. At precision 0.016 the
orchard layer of that map is worse than useless — an inspector acting on it
would be wrong 98 times in 100. Macro F1 scored this as an improvement.

**Concrete edge case.** Suppose alpha for orchards is raised further. Recall
keeps climbing, macro F1 keeps improving, and the orchard layer of the map
keeps degrading, because precision falls faster than recall rises for a class
with a 0.5% base rate. The loss and the metric agree with each other and both
disagree with the task. This is not a tuning failure; it is the metric
measuring the wrong thing.

**Monitoring and mitigation.** Three options, in increasing cost:

1. Report per-class precision alongside macro F1 and set a precision floor
   below which a class is reported as "unmapped" rather than predicted. This
   is cheap and honest, and it is what I would actually ship.
2. Weight the evaluation by parcel area rather than parcel count, since the
   task is about hectares mapped, not parcels counted. Not implemented here:
   the feature table carries `field_id` but not geometry, and recovering area
   means joining back to the RPG parcel shapefiles. Flagged as the highest
   value unfinished item.
3. Replace the classification objective with an explicitly cost-sensitive one,
   where confusing two cereals costs less than confusing a cereal with an
   orchard. This is the principled fix and the natural extension of the
   custom-objective work already done, since the machinery for supplying a
   custom gradient is already in place.

---

## Entry 9 — LORO cross validation, and a retraction

The Entry 7 experiment held out one region. That is n=1, and I drew a
conclusion from it that does not hold. This entry records the correction.

**Protocol.** Leave-one-region-out over all four NUTS-3 regions. For every
fold, a size-matched random split is drawn from the same pool with identical
train and test sizes, so the only difference between the two arms is how the
split was constructed rather than how much data each arm received. A
majority-class baseline is fitted on train and scored on test per fold.

(15% subsample; full-data run pending. Numbers below are indicative.)

**Baseline.** Majority-class accuracy 0.2994 +/- 0.0190. Regional accuracy of
~0.547 is therefore about 1.8x baseline, which is the context the earlier
entries were missing.

**Main result, now with error bars:**

| objective | split | accuracy | macro F1 |
|---|---|---|---|
| softmax | random | 0.5686 +/- 0.0030 | 0.4032 +/- 0.0040 |
| softmax | regional | 0.5470 +/- 0.0048 | 0.3784 +/- 0.0118 |
| softmax | **gap** | **+0.0216 +/- 0.0067** | **+0.0248 +/- 0.0113** |
| focal | random | 0.5631 +/- 0.0030 | 0.4517 +/- 0.0038 |
| focal | regional | 0.5387 +/- 0.0122 | 0.4216 +/- 0.0107 |
| focal | **gap** | **+0.0244 +/- 0.0147** | **+0.0300 +/- 0.0098** |

**The leakage gap is real and robust.** Positive in every fold, for both
objectives and both metrics. Modest in size — about two accuracy points — but
consistent. This is the defensible version of the headline claim.

**RETRACTION of the Entry 7 claim.** Entry 7 stated that the leakage gap
roughly doubles under focal loss, based on frh04 alone (softmax +0.0184,
focal +0.0389). Across four folds the mean gaps are +0.0216 and +0.0244,
differing by well under one standard deviation. The per-fold breakdown shows
why the original reading was wrong:

| fold | softmax gap | focal gap |
|---|---|---|
| frh01 | +0.0270 | +0.0255 |
| frh02 | +0.0158 | +0.0039 |
| frh03 | +0.0279 | +0.0382 |
| frh04 | +0.0157 | +0.0302 |

Focal's gap is larger on two folds and smaller on two. frh04 — the region I
happened to pick first — is one where it is larger, and I generalised from
it. The effect is not there.

**A weaker claim that the data does support.** Focal does not inflate the
mean gap, but it does inflate the variance. Gap std rises 0.0067 -> 0.0147,
and regional accuracy std rises 0.0048 -> 0.0122. So reweighting toward rare
classes makes cross-region transfer *less predictable* without making it
worse on average. The practical consequence is specific: a single held-out
region is a poor validation signal for a reweighted model, because the
fold-to-fold spread is roughly double.

I am deliberately keeping both the original claim and its retraction in this
log rather than quietly editing Entry 7. The sequence is the evidence that
the validation protocol was doing work.

**Per-class, regional folds, mean over folds:**

| class | softmax | focal |
|---|---|---|
| barley | 0.143 | 0.289 |
| wheat | 0.453 | 0.500 |
| rapeseed | 0.410 | 0.472 |
| corn | 0.714 | 0.721 |
| orchards | 0.000 | 0.000 |
| permanent meadows | 0.331 | 0.427 |
| temporary meadows | 0.599 | 0.543 |

Focal's benefit is real and concentrated where predicted: barley roughly
doubles, permanent meadows and rapeseed gain substantially, and the cost is
paid almost entirely by temporary meadows. Corn is unaffected.

**Orchards correction.** At this subsample orchards is 0.000 under *both*
objectives. The 0.029 reported in Entry 7 was from the full dataset and is
marginal enough that it did not survive subsampling. The Entry 8 argument
about precision 0.016 still stands as an illustration of the loss/objective
mismatch, but it should be presented as a full-data observation rather than a
stable property, pending the full LORO run.

**Known remaining leak.** `alpha` is computed from the full pool rather than
per fold from the training regions only. Class proportions are stable across
these four regions so this should not change conclusions, but it is a genuine
if small violation of fold independence, and I would recompute it per fold in
a version I was publishing.

---

## Entry 10 — Order-aware features

Entry 5 left a hypothesis open: the modest leakage gap might be an artefact
of the v1 features being ORDER-FREE. Permute the 45 acquisition dates and
mean/std/min/max are unchanged, so the model cannot represent phenology at
all — and phenology timing, not spectral magnitude, is what separates wheat
from barley. A model that never sees local temporal structure cannot overfit
to it.

**Three order-aware families added** (`src/features_v2.py`, 202 features):

| family | n | what it captures |
|---|---|---|
| basic (v1) | 52 | order-free magnitude summary |
| segment means | 78 | coarse temporal shape, 6 windows per band |
| harmonics | 65 | smooth seasonal shape, order-2 Fourier per band |
| NDVI phenology | 7 | peak height, peak timing, green-up and senescence slopes, integral |

**Verification before use**, same discipline as the focal loss:

1. *Shuffle test.* Permuting timesteps leaves `basic` unchanged (max delta
   2.2e-16) and changes every other family materially (0.23 to 0.36). This
   mechanically proves the families are what they claim to be, rather than
   my asserting it.
2. *Harmonic recovery.* Fitted against a planted signal
   `0.5 + 0.3cos - 0.2sin`, the coefficients come back as 0.5000, 0.3000,
   -0.2000 with second-order terms at 1e-17.
3. *Synthetic phenology.* On a constructed season the NDVI peak lands at
   exactly mid-series with positive green-up and negative senescence.

**Band-order assumption, and how it was checked.** I assumed standard
Sentinel-2 ordering, so red = index 3 and NIR = index 7. I did not read this
off the data, so I wrote a falsifier rather than trusting it: real vegetated
parcels must show clearly positive NDVI peaking mid-season. Observed on 500
frh04 parcels: peak NDVI median 0.657 (5th-95th 0.587-0.731), peak timing
median 0.466. Tight, positive, mid-season. Swapped indices would give
strongly negative values; two SWIR bands would give noise near zero. Neither
occurred, so the assumption holds. Rebuild produced 608,263 x 205 with zero
NaNs and zero infs.

**Result: a large accuracy gain, separate from the leakage question.**

| feature set | n | regional acc | random acc | gap |
|---|---|---|---|---|
| basic (order-free) | 52 | 0.5450 | 0.5675 | +0.0225 |
| basic + segment | 130 | 0.6388 | 0.6642 | +0.0254 |
| basic + harmonic | 117 | 0.6381 | 0.6605 | +0.0224 |
| basic + NDVI phenology | 59 | 0.5427 | 0.5722 | +0.0295 |
| all | 202 | 0.6365 | 0.6650 | +0.0286 |

Regional accuracy rises from 0.5450 to 0.6388, **+9.4 points**, essentially
all of it from segment means or harmonics. Temporal ordering matters enormously
for the task itself, which retrospectively makes the v1 featurisation look
like a much weaker baseline than it seemed at the time.

---

## Entry 11 — A statistical error, and what it hid

**The error.** My first pass tested each feature set's mean gap against the
baseline using the pooled spread across folds, and reported "inconclusive"
for everything. That test is wrong. The same four regions appear in every
feature set, so the observations are PAIRED. An unpaired comparison throws
away the pairing and is severely underpowered at n=4.

**Corrected, paired against the baseline:**

| feature set | d(gap) | t | p | folds agreeing |
|---|---|---|---|---|
| + segment | +0.0029 | 0.32 | 0.771 | 2/4 |
| + harmonic | -0.0001 | -0.01 | 0.995 | 2/4 |
| **+ NDVI phenology** | **+0.0070** | **4.21** | **0.025** | **4/4** |
| + all | +0.0061 | 0.52 | 0.637 | 2/4 |

**The hypothesis is supported, but only for the phenology features.** Adding
seven NDVI curve descriptors widens the gap in all four folds, p = 0.025.
The mechanism is visible in the components: those features *raise* random-split
accuracy (0.5675 -> 0.5722) while *lowering* regional accuracy
(0.5450 -> 0.5427). They help when train and test share regions and hurt when
they do not. That is the Entry 5 hypothesis isolated almost surgically —
phenology timing is locally varying signal, the model exploits it, and it
does not transfer.

**The bulk temporal features behave differently, and that is informative.**
Segment means and harmonics buy 9 points of accuracy without reliably
widening the gap. So they are adding genuinely transferable structure, not
local quirks. The naive reading — "richer features means more leakage" — is
wrong. It depends on *which* structure the features expose.

With 3 degrees of freedom these p-values are indicative, not strong evidence.
The 4/4 sign pattern is the more robust signal and is reported alongside.

**The variance finding replicates.** Standard deviation of the gap across
folds:

| feature set | gap std | vs baseline |
|---|---|---|
| basic | 0.0036 | 1.0x |
| + NDVI | 0.0055 | 1.5x |
| + harmonic | 0.0151 | 4.2x |
| + segment | 0.0173 | 4.8x |
| all | 0.0222 | 6.2x |

Entry 9 found the same pattern for focal loss (0.0067 -> 0.0147). Two
unrelated interventions — reweighting the objective toward rare classes, and
exposing temporal structure — both inflate the *variance* of cross-region
transfer without reliably moving its mean. This is now the most robust
finding in the project, supported by two independent experiments:

> Increasing a model's capacity to exploit local structure makes
> cross-region generalisation less *predictable*, not merely worse.

The practical consequence is concrete: single-region validation becomes an
unreliable guide exactly as models get more sophisticated, because the
fold-to-fold spread grows by 4-6x while the mean barely moves.

**frh04 is an outlier, and this matters for the benchmark.** Per-fold gaps
with temporal features: frh04 collapses to +0.0046 (segment), +0.0011
(harmonic), -0.0005 (all), while frh02 and frh03 rise to +0.04/+0.05. frh04
also records the highest regional accuracy of any fold (0.664). So frh04 is
the one region where cross-region transfer is essentially free — and it is
the region the BreizhCrops benchmark designates for evaluation. A protocol
evaluated only on frh04 would conclude that spatial generalisation is a
non-issue for this dataset. Three of the four folds say otherwise.

---

## Entry 12 — Full-data confirmation

Entries 9-11 were run on a 15% stratified subsample for speed. Both
experiments were rerun on the full 608,195 parcels. This entry records what
held, what strengthened, and what needed correcting.

**Every directional conclusion replicated. None flipped.** That is itself
worth recording: it validates using the subsample for iteration, since a
protocol that routinely reversed between 15% and 100% would have made all the
earlier development work untrustworthy.

**LORO, focal vs softmax (full data):**

| objective | split | accuracy | macro F1 |
|---|---|---|---|
| softmax | random | 0.5905 +/- 0.0005 | 0.4384 +/- 0.0009 |
| softmax | regional | 0.5679 +/- 0.0046 | 0.4099 +/- 0.0154 |
| softmax | gap | +0.0226 +/- 0.0049 | +0.0286 +/- 0.0148 |
| focal | random | 0.5758 +/- 0.0009 | 0.4834 +/- 0.0008 |
| focal | regional | 0.5497 +/- 0.0212 | 0.4527 +/- 0.0212 |
| focal | gap | +0.0261 +/- 0.0210 | +0.0307 +/- 0.0216 |

Majority baseline 0.2998 +/- 0.0171, so regional softmax accuracy is 1.9x
baseline.

- *Leakage gap:* +0.0226 +/- 0.0049, positive in all four folds. Holds.
- *Entry 9 retraction:* focal does not raise the mean gap (+0.0226 vs
  +0.0261, overlapping). The retraction stands.
- *Variance under focal:* gap std 0.0049 -> 0.0210, a **4.3x** increase.
  Stronger than the 2.2x seen at 15%.

**New observation, and the sharpest single point in the project.** Random-split
accuracy varies across folds by +/- 0.0005 (softmax) and +/- 0.0009 (focal).
Regional accuracy varies by +/- 0.0046 and +/- 0.0212. So random cross
validation does not merely overstate the mean — it conceals the variance
almost completely. A practitioner validating a focal-loss model with random
CV would see a model stable to the fourth decimal place, and would have no
signal at all that its performance on a new region could swing by several
points depending on which region it was.

**Feature sets (full data):**

| feature set | n | regional | random | gap | gap std |
|---|---|---|---|---|---|
| basic | 52 | 0.5680 | 0.5920 | +0.0240 | 0.0037 |
| + segment | 130 | 0.6575 | 0.6837 | +0.0262 | 0.0178 |
| + harmonic | 117 | 0.6565 | 0.6838 | +0.0273 | 0.0176 |
| + NDVI | 59 | 0.5638 | 0.5965 | +0.0327 | 0.0073 |
| all | 202 | 0.6567 | 0.6878 | +0.0311 | 0.0233 |

Paired against basic:

| feature set | d(gap) | t | p | folds |
|---|---|---|---|---|
| + segment | +0.0022 | 0.28 | 0.800 | 2/4 |
| + harmonic | +0.0032 | 0.44 | 0.691 | 2/4 |
| **+ NDVI** | **+0.0086** | **4.69** | **0.018** | **4/4** |
| all | +0.0070 | 0.69 | 0.541 | 2/4 |

- *NDVI widens the gap:* p = 0.018, 4/4 folds. Stronger than at 15%
  (p = 0.024). The mechanism holds exactly: random accuracy +0.0045,
  regional accuracy -0.0042.
- *Accuracy gain:* 0.5680 -> 0.6575, +9.0 points. Holds.
- *Variance growth:* 0.0037 -> 0.0233, **6.4x**. Holds.

**Corrections to earlier entries:**

1. *frh04 (Entry 11).* At 15% the full-temporal gap for frh04 was -0.0005,
   which I described as collapsing "to zero or negative". At full size it is
   +0.0038, and +0.0083 / +0.0037 for segment and harmonic. The claim should
   read *near zero*, not negative. The substance survives: frh04 remains a
   clear outlier with the highest regional accuracy of any fold (0.6837), and
   the benchmark's designated test region is still the one where transfer is
   nearly free.
2. *Orchards (Entries 7 and 9).* Entry 9 reported orchards at 0.000 under
   both objectives and cautioned that the full-data 0.029 was marginal. The
   full LORO gives **0.036** under focal, 0.000 under softmax. So focal does
   lift orchards off zero at full size, just not by much, and 15% was too
   small a sample to resolve it — 3,070 orchard parcels shrink to about 460,
   spread over four test folds. The Entry 8 argument (precision far below
   recall, so the orchard layer of any map is unusable) stands.
3. *Rapeseed (Entry 9).* At 15% focal lifted rapeseed from 0.410 to 0.472.
   At full size: 0.521 vs 0.519. That gain did not replicate and should not
   be claimed.

**Per-class, regional LORO, full data:**

| class | softmax | focal |
|---|---|---|
| barley | 0.180 | 0.341 |
| wheat | 0.480 | 0.536 |
| rapeseed | 0.521 | 0.519 |
| corn | 0.733 | 0.744 |
| orchards | 0.000 | 0.036 |
| permanent meadows | 0.340 | 0.453 |
| temporary meadows | 0.615 | 0.541 |

**Summary of findings, in order of strength:**

1. Richer models make cross-region transfer less *predictable*: gap variance
   rises 4.3x under focal loss and 6.4x under temporal features, while the
   mean gap barely moves. Two independent interventions, same pattern.
2. Random cross validation hides this almost entirely (+/- 0.0005 vs
   +/- 0.0212).
3. NDVI phenology features specifically widen the leakage gap, 4/4 folds,
   p = 0.018, with a visible mechanism.
4. The leakage gap itself is real but modest: +0.0226 +/- 0.0049.
5. Temporal features add 9 points of accuracy; focal loss trades about 2
   points of accuracy for 4 points of macro F1.

---

## Entry 13 — Final model, the gamma ablation, and where the error lives

**Design.** All 202 features, evaluated with nested leave-one-region-out.
Gamma selected per outer fold from {0, 1, 2, 3} by an *inner* LORO over the
three training regions, on macro F1. Inner LORO rather than a single
validation region because Entries 9-12 show single-region validation is
unreliable for exactly this kind of model. Class weights alpha now computed
per fold from training regions only, closing the leak flagged in Entry 9.
Selection used a 30% stratified subsample of each inner training set for
cost; final models were fitted on the full training regions.

### The ablation: gamma = 0 in every fold

| outer fold | g=0 | g=1 | g=2 | g=3 |
|---|---|---|---|---|
| frh01 | **0.5129** | 0.5089 | 0.5063 | 0.5055 |
| frh02 | **0.5169** | 0.5148 | 0.5137 | 0.5121 |
| frh03 | **0.5474** | 0.5451 | 0.5439 | 0.5399 |
| frh04 | **0.5121** | 0.5083 | 0.5058 | 0.5034 |

(inner macro F1, mean over three inner folds)

Gamma = 0 wins in all four folds, and inner F1 falls monotonically as gamma
rises in all four. The differences are small, but the direction is perfectly
consistent. At gamma = 0 focal loss reduces to **class-weighted cross
entropy**: the alpha weights remain and the focusing term vanishes.

**Correction to Entries 6-9.** The macro F1 gains attributed to "focal loss"
throughout those entries came from the class weights, not from the focusing
term. The focusing term contributes nothing measurable and slightly hurts.
Entry 6's mechanism ("down-weights well-classified samples, concentrating the
gradient on hard examples") describes what gamma > 0 does, and gamma > 0 turns
out not to help here. The accurate claim: *inverse-frequency reweighting* helps
rare classes; *focusing* does not add to it.

**Why the custom objective was still necessary.** The obvious question is why
build a focal objective if gamma = 0 wins. Because the ablation required the
ability to vary gamma. Without the custom gradient I could not have tested
the focusing term, and would have credited focal loss on the strength of
Entry 7 alone. The derivation and verification (Entry 6) stand; what changed
is which part of the loss earns the result. At gamma = 0 the objective is
mathematically equivalent to built-in softmax with per-sample weights
`alpha[y]`, which would be a further cheap correctness check.

### Final comparison (fold mean +/- sd, 4 held-out regions)

| model | accuracy | macro F1 |
|---|---|---|
| softmax, 202 features | 0.6567 +/- 0.0228 | 0.5187 +/- 0.0365 |
| class-weighted CE, 202 features | 0.6457 +/- 0.0420 | 0.5480 +/- 0.0652 |

Per fold, weighted minus softmax:

| fold | d accuracy | d macro F1 |
|---|---|---|
| frh01 | -0.0004 | +0.0516 |
| frh02 | **+0.0300** | +0.0675 |
| frh03 | **-0.0497** | -0.0198 |
| frh04 | -0.0241 | +0.0180 |

**The variance finding, a third time and in its starkest form.** The same
intervention gains 3 accuracy points on frh02 and loses 5 on frh03, an 8-point
swing, and on frh03 it loses on macro F1 too. The fold-mean summary ("trades
about 1 point of accuracy for 3 of F1") is true and hides all of this.
Accuracy sd roughly doubles (0.0228 -> 0.0420). Three independent
interventions (reweighting, temporal features, and reweighting again on top
of temporal features) now show the same pattern.

The inner selection shows it too: inner macro F1 sd of 0.04-0.07 across
inner regions, against gamma differences of under 0.01. Region-to-region
variation dwarfs the effect being tuned.

**Reporting note.** Fold means above weight each region equally. The confusion
figure reports *pooled* metrics over all 608,195 parcels (softmax 0.656 /
0.517, weighted 0.644 / 0.539), which weight regions by size. They differ
slightly for that reason. The fold mean +/- sd is the primary result; pooled
numbers are used only for the confusion analysis.

### Per-class precision and recall (pooled)

| class | softmax P / R / F1 | weighted P / R / F1 |
|---|---|---|
| barley | 0.540 / 0.233 / 0.326 | 0.379 / 0.599 / 0.465 |
| wheat | 0.700 / 0.792 / 0.743 | 0.735 / 0.759 / 0.747 |
| rapeseed | 0.675 / 0.668 / 0.671 | 0.459 / 0.838 / 0.593 |
| corn | 0.868 / 0.903 / 0.885 | 0.899 / 0.891 / 0.895 |
| orchards | 0.000 / 0.000 / 0.000 | 0.028 / 0.081 / 0.042 |
| perm. meadows | 0.494 / 0.292 / 0.367 | 0.490 / 0.451 / 0.470 |
| temp. meadows | 0.553 / 0.731 / 0.629 | 0.622 / 0.518 / 0.565 |

Reweighting is a precision-for-recall trade, not a free improvement. Barley
recall rises 0.233 -> 0.599 while precision falls 0.540 -> 0.379; rapeseed
recall rises to 0.838 while precision falls to 0.459, and rapeseed F1 goes
*down*.

### Where the error actually lives: the meadow distinction

**Half of all errors are one confusion.** Permanent <-> temporary meadow
errors account for **51.9%** of all softmax errors (108,631 of 209,486) and
46.5% under reweighting. Merging the two classes:

| model | 7-class accuracy | meadows merged |
|---|---|---|
| softmax | 0.6556 | **0.8342** (+0.179) |
| weighted CE | 0.6439 | 0.8095 (+0.166) |

**Why: the label is defined by information the input does not contain.**
Under EU CAP rules, permanent grassland is land in grass that has not been
included in the holding's crop rotation for five years or more (Regulation
(EU) No 1307/2013, Art. 4(1)(h), as amended by Regulation (EU) 2017/2393).
Temporary grassland is grass for five years or less. BreizhCrops labels are
derived from farmers' CAP parcel declarations, so they follow this
definition. The distinction is *multi-year land-use history*. The model sees
a single 2017 season.

It is not completely unobservable. Temporary grassland tends to be more
intensively managed and is sometimes resown, which leaves traces in a
single season, which is why meadow recall is well above chance. But the
defining criterion itself is outside the input. A large share of the
remaining error is therefore a property of the task specification, not
something a better model or loss could remove.

This reframes the headline. "0.66 accuracy" sounds mediocre; "0.83, with
the bulk of the residual error coming from a label defined by five years of
history the model never observes" is both more accurate and more useful.
It is also the single clearest example in the project of the loss/objective
gap from Entry 8: cross entropy penalises the meadow confusion as heavily as
confusing corn with orchards, even though one distinction is observable and
the other largely is not.

**Future work this directly motivates:** multi-year input (several seasons
per parcel), or the previous year's declared crop as a feature.

### Orchards look like grassland

Under reweighting, 8,931 parcels are predicted as orchards and 250 are right:
**precision 0.028**. Of the false positives, 7,152 (80%) are meadows
(3,252 permanent, 3,900 temporary). Going the other way, softmax sends 79% of
true orchards to one of the two meadow classes.

Plausible explanation, stated as a hypothesis rather than established: orchard
parcels typically carry grass cover between the trees, so averaged over a
whole parcel at Sentinel-2 resolution they are spectrally close to grassland.
This is consistent with the confusion structure but I have not verified it
against orchard management practice in Brittany.

The Entry 8 argument now has its exact numbers: macro F1 records orchards
rising from 0.000 to 0.042 as an improvement, while in map terms the
reweighted model paints 8,681 parcels as orchards that are not.

**Barley <-> wheat** is the other substantive confusion (20% of barley
predicted as wheat, 15% of wheat as barley under reweighting). Both are
winter cereals with similar phenology, which is also why barley was the class
the temporal features and reweighting helped most.

### Which model I would deploy

Softmax with all 202 temporal features. It is about half as variable across
regions (accuracy sd 0.0228 vs 0.0420), and given that the project's central
finding is that cross-region transfer is unpredictable, choosing the more
predictable model is a principled decision rather than a hedge. It also
avoids the precision collapse on rapeseed and orchards. The reweighted model
is the right choice only if rare-class recall matters more than map
precision, for example screening parcels for a human to inspect.

---

## Entry 14 — The notebook

`crop_leakage.ipynb` is the canonical, self-contained version of the
pipeline: environment setup, data download, feature extraction, the focal
objective, and the three experiments behind the reported results. The
scripts in `scripts/` remain as the development history this log describes.

Two modes: `quick` (all four regions, 5% of parcels, 50 rounds) to check the
pipeline runs, and `full` (all parcels, 200 rounds) for the reported numbers.
Both have to download all four regions: the research question is about
generalising across regions, and nested selection needs at least three, so
the download cannot be reduced without changing the question.

**Differences from the scripts, which may move numbers in the third decimal:**

1. Experiment 1 computes class weights per fold from training data only. The
   original LORO script used weights from the full pool (the leak flagged in
   Entry 9).
2. All experiments use the float32 feature table. The first LORO run used the
   float64 v1 table, and XGBoost's histogram binning can differ very slightly
   between the two.

The journal reports the notebook's full-mode numbers, since that is the
archived, reproducible version. Conclusions are rechecked against them after
the full run.

**Bug found while building it.** The 52 basic features were computed stat by
stat (all 13 band means, then all 13 stds, and so on) but *named* band by
band. So the column labelled `b0_std` held band 1's mean. It is visible in the
first `head()` output, where `b0_min` (0.486) exceeds `b0_max` (0.192). No
result is affected: every experiment selected whole feature families by
position, and XGBoost ignores column names. It would have mattered for any
per-feature interpretation, such as feature importance. Fixed in the
notebook, which now includes a test that constructs input with known
per-band values and checks that each named feature holds the right one.

---

## Entry 15 — Cloud contamination: a limitation, stated rather than fixed

BreizhCrops is Sentinel-2 **Level-1C**, top-of-atmosphere reflectance. No
atmospheric correction, no cloud mask. Every acquisition is present whether
or not the sky was clear, which is why some parcel reflectance values exceed
1.0 (visible in the value-range printout in the notebook's data section).

**Effect on the features.** Cloud raises reflectance across most bands and
suppresses NDVI, because cloud is bright in red as well as near-infrared.
Exposure varies by feature:

- `max` and `*_peak` are the most exposed: a single cloudy date sets them.
- `mean` and segment means are diluted but not immune.
- Harmonic coefficients are the most robust, since a least-squares fit over
  45 points damps isolated spikes.

**Why it was not fixed.** Proper masking uses Sentinel-2's scene
classification layer or cloud-probability band, and neither survives into
BreizhCrops, which supplies parcel-averaged reflectance only. The remaining
option is a reflectance threshold heuristic, which has a real cost: dropping
dates leaves parcels with different numbers of observations, and the
fixed-length featurisation (6 segment windows, a 45-point harmonic fit)
assumes a common grid. Handling that properly means interpolating onto a
regular calendar first, which is a project of its own.

**The part that bears on the main finding.** Cloud is *spatially* correlated:
a cloudy day covers a whole area, not scattered individual parcels. So cloud
contamination is partly a regional effect. A model trained on three regions
may be fitting their cloud patterns as well as their agronomy, which means an
unknown share of the measured leakage gap could be cloud rather than crop
phenology. This does not invalidate the gap, which is measured identically in
both arms, but it does mean "regional difference" is not purely agronomic.

Distinguishing the two would need a cloud-free comparison, and the data as
distributed does not permit one. This is the most substantive untested
confound in the project and belongs in the limitations section.

---

## Entry 16 — Full notebook run: replication, and two new findings

`crop_leakage.ipynb` run in full mode: 608,195 parcels, 7 classes, 219
features, 400 minutes, zero errors. This is the canonical result set and the
one the report quotes.

### Everything replicated

| claim | scripts | notebook |
|---|---|---|
| leakage gap | +0.0226 +/- 0.0049, 4/4 folds | **+0.0225 +/- 0.0043, 4/4** |
| random sd vs regional sd | 0.0005 vs 0.0046 | **0.0006 vs 0.0039 (6.9x)** |
| gap sd, basic -> all features | 6.4x | **5.2x** |
| temporal feature gain | +9.0 points | **+9.5 points** |
| meadow share of all errors | 51.9% | **51.9%** |
| accuracy with meadows merged | 0.834 | **0.837** |
| gamma selected | 0 in every fold | **0 in every fold** |

Independent reimplementation, per-fold class weights instead of pooled, 219
features instead of 202, float32 instead of float64. Nothing moved. The
gamma=0 ablation in particular now has two independent confirmations.

Majority baseline 0.2998; final model 0.6625, so 2.2x baseline.

### Finding refined: it is greenness phenology, not indices in general

| index | built on | d(gap) | p | folds |
|---|---|---|---|---|
| **NDVI** | red / NIR | **+0.0100** | **0.019** | **4/4** |
| **EVI** | red / NIR (+blue) | **+0.0092** | 0.067 | **4/4** |
| NDRE | red edge | +0.0016 | 0.087 | 3/4 |
| NDWI | SWIR | +0.0020 | 0.682 | 3/4 |

Adding NDRE, NDWI and EVI was done specifically to test whether the earlier
NDVI result was about NDVI or about index features generally (the same n=1
error as Entry 7, caught before it was repeated). The answer is neither of
the obvious ones:

The two **red/NIR greenness** indices behave almost identically (+0.010, 4/4
folds each). The indices built on other physics — chlorophyll via red edge,
water content via SWIR — barely move the gap. So the claim is narrower and
more mechanistic than "index features leak": **red/NIR greenness phenology is
the component that does not transfer across regions.** Green-up and
senescence timing depend on local soil and weather; chlorophyll and water
content do not encode timing the same way.

Caveat on the strength of this: NDVI and EVI are strongly correlated, both
being red/NIR contrasts. This is one mechanism confirmed in two formulations,
not two independent confirmations.

**Unexpected practical result: NDWI is the most useful index.** It raises
regional accuracy more than any other (0.5673 -> 0.5833, +1.6 points) while
leaving the gap essentially unchanged. So water-content features help
generalisation, while greenness features help the random split and hurt
transfer. For anyone building a crop map to deploy on unseen regions, that is
an actionable recommendation.

### Overfitting: the model is capacity-limited, not memorising

Learning curves on frh01, all 219 features, 200 rounds:

| | accuracy at round 200 |
|---|---|
| train | 0.7357 |
| random test | 0.7169 |
| held-out region | 0.6722 |

Train-to-regional gap is 0.064, and training accuracy was **still rising** in
the last rounds (0.7349 -> 0.7357 over rounds 197-199). The model is not
overfitting at this depth and round count; more capacity would likely help
slightly.

This is consistent with the meadow result. The model cannot fit the *training*
data well either, because a large share of the task is not determined by the
input at all. A model that cannot reach 0.75 on data it has seen is not one
whose errors are mostly variance.

### Feature importance, and an empirical version of the cloud confound

Grouped permutation importance, mean accuracy drop over four folds:

| group | regional | random | band |
|---|---|---|---|
| band 2 | 0.1805 | 0.1997 | B3, green |
| band 12 | 0.1195 | 0.1378 | B12, SWIR-2 |
| **band 9** | **0.1066** | 0.1120 | **B9, water vapour 945nm** |
| band 6 | 0.0798 | 0.0839 | B7, red edge |
| band 11 | 0.0656 | 0.0752 | B11, SWIR-1 |
| index ndwi | 0.0632 | 0.0722 | - |
| **band 10** | **0.0499** | 0.0489 | **B10, cirrus 1375nm** |

**B9 and B10 are atmospheric sounding bands.** B10 (cirrus, 1375 nm) is
strongly absorbed by water vapour and is designed so that almost no surface
signal reaches the sensor; it exists to detect high cloud, not ground cover.
Shuffling it costs 5 accuracy points, and B9 costs 10.7.

The model is therefore using **atmospheric state** as a predictor of crop
type. Entry 15 argued on physical grounds that cloud contamination was a
plausible confound; this measures it. Since atmospheric conditions on a given
acquisition date are spatially correlated, part of what the model learns as
"region" is weather rather than agronomy. That is a real qualification on the
leakage result and belongs in the discussion.

It does not invalidate the gap — both arms see the same data — but it means
"regional difference" is not purely agronomic, and the gap is not purely a
crop-phenology effect.

**Reading the low-importance groups correctly.** Red (band 3, drop 0.0005)
and NIR (band 7, 0.0225) look nearly irrelevant, which is not what it seems.
Shuffling `b3_*` destroys red's raw statistics but leaves NDVI and EVI intact,
and those carry the same information. Permutation importance measures
reliance *given everything else present*, not intrinsic usefulness. The same
caution applies in reverse to B9 and B10: they are not substitutable by
anything else in the feature set, which is part of why their drop is large.

**Obvious next experiment**, not run: drop B9 and B10 and retrain. If accuracy
falls materially, the model was depending on atmosphere; if the gap narrows,
part of the measured leakage was atmospheric rather than agronomic. Eight
fits, about an hour.

---

## Final summary of findings

All numbers from the full notebook run (608,195 parcels, 219 features,
leave-one-region-out over four NUTS-3 regions). Majority baseline 0.2998.

1. **Increasing a model's capacity to exploit local structure makes
   cross-region transfer less predictable.** Fold-to-fold spread of the
   leakage gap rises 5.2x from basic to full features, and the spread of
   regional accuracy roughly doubles under class reweighting, while the mean
   effects stay small. Three interventions, one pattern.

2. **Random cross validation conceals this almost entirely.** Accuracy across
   folds varies by +/- 0.0006 under random splits and +/- 0.0039 under
   regional splits, 6.9x more. A practitioner validating the usual way sees a
   model stable to the fourth decimal and gets no warning.

3. **About half the remaining error is not the model's to fix.** Permanent
   versus temporary meadow confusion is 51.9% of all errors; merging those two
   classes lifts accuracy from 0.661 to 0.837. The distinction is defined by
   five years of land-use history (Reg. (EU) 1307/2013 Art. 4(1)(h)), and the
   model sees one season.

4. **It is red/NIR greenness phenology specifically that fails to transfer.**
   NDVI (+0.0100, p=0.019) and EVI (+0.0092) widen the gap in 4/4 folds;
   red-edge and SWIR indices do not. NDWI meanwhile gives the largest gain in
   regional accuracy (+1.6 points) of any index.

5. **The leakage gap is real and modest**: +0.0225 +/- 0.0043, positive in
   every fold.

6. **The model relies on atmospheric bands.** B9 (water vapour) and B10
   (cirrus) carry permutation importance of 0.107 and 0.050 despite carrying
   little surface signal, so an unquantified share of the regional effect is
   weather rather than agronomy.

7. **Not overfitting.** Train 0.736 against held-out region 0.672, with
   training accuracy still rising at round 200. The model is capacity-limited,
   which is consistent with finding 3.

8. **Component contributions.** Temporal features add 9.5 accuracy points.
   Class reweighting trades precision for recall (macro F1 0.531 -> 0.561,
   accuracy 0.663 -> 0.657). The focal focusing term adds nothing: gamma = 0
   was selected in every fold.

---

## Use of AI tools

Claude (Anthropic) was used extensively throughout, as a collaborator on
design, code and analysis. Entries above are written in the first person for
readability; this section states plainly who originated what, because that is
what I need to be able to defend.

**Originated by me:**
- The project direction: Option 2, crop type classification, XGBoost.
- The decision to treat evaluation protocol, rather than accuracy, as the
  object of study.
- The principle that evaluation should split by spatial group rather than at
  random. This comes from commercial crop classification work, where the
  requirement was that test paddocks be distinctly different from training
  paddocks, and that no training paddock sat close to or inside the area being
  tested, so that the evaluation carried as little optimistic bias as possible.
- The decision to add NDRE, NDWI and EVI, and to test them as separate feature
  sets rather than as one block. That separation is what allowed the finding to
  be narrowed from "index features" to "red/NIR greenness phenology".
- The request for grouped feature importance, to assess whether individual
  bands could be dropped. This is what surfaced the model's reliance on the
  atmospheric sounding bands.
- The insistence on measuring runtime empirically before committing to a long
  run, and on progress instrumentation so a silent headless run could be
  monitored.
- The decision to run every reported result at full dataset scale rather than
  accept subsample figures, and to finish the analysis before writing up.
- Running every experiment and reading every output.

**Originated by Claude, then reviewed and run by me:**
- Dataset choice (BreizhCrops).
- Project structure and effectively all of the code.
- The focal loss derivation, its implementation, and the two verification
  strategies for it.
- The formalisation of the spatial-splitting principle into the protocol
  actually used: leaving out each of the four regions in turn, the
  size-matched random control, and the nested inner leave-one-region-out for
  gamma selection. Also the choice of paired testing.
- The v2 temporal features and the figures.
- Most of the interpretation drafted in these entries, and the first drafts of
  the report.

**AI errors caught during the project.** This is the most useful record of
critical review, because each one would have gone into the report unchecked:

1. *Generalising from one fold (Entry 7 -> Entry 9).* Claude claimed the
   leakage gap "roughly doubles" under focal loss, from a single held-out
   region. Four-fold LORO showed no such effect. Retracted.
2. *Wrong statistical test (Entry 11).* Claude's first comparison of
   feature sets used an unpaired test on paired fold data and reported
   everything as inconclusive. The paired test found a significant NDVI
   effect (p = 0.018). The first version would have hidden the project's
   cleanest mechanistic result.
3. *Crediting the wrong component (Entry 13).* Claude proposed focal loss
   as the fix for rare classes and attributed the resulting gains to the
   focusing term. The nested ablation selected gamma = 0 in every fold: the
   gains came from the class weights.
4. *Hard-coded figure claims.* Figure titles were first written with fixed
   numbers ("+9 points", "~6x") that could have contradicted the full-data
   results. Rewritten to be computed from the data. The confusion matrix was
   also titled "focal" after gamma = 0 was selected, and was corrected.
5. *Inconsistent reporting.* A variance ratio was written as 6.3x in the log
   (from rounded values) while the figure showed 6.4x (from unrounded
   values). Corrected to match.
6. *Mislabelled features.* The basic feature names did not match the
   order of their values (Entry 14). No result was affected, but any
   feature-level interpretation would have been wrong. Found while building
   the notebook; a naming test now guards it.
7. *Conclusions baked into figure titles, a second time.* After the fixed
   numbers were removed, titles still asserted conclusions in words ("mean
   gap barely moves"). Testing the notebook on synthetic data, where the mean
   gap did move, exposed it. Titles now describe the plotted data only.
8. *An overstated AI-use section.* An earlier draft of this section said I
   had directed the dataset choice and research question. I had not.
   Corrected to this version.

**How claims were verified rather than trusted:**
- Focal loss derivatives checked against finite differences (gradient to
  1e-10) and against the closed-form gamma = 0 reduction (exact, 0.0).
- Temporal features checked with a timestep-shuffle test, planted-signal
  harmonic recovery, and a synthetic phenology curve.
- The Sentinel-2 band-order assumption falsified-or-confirmed against real
  NDVI distributions before the rebuild.
- Every subsample result rerun at full size (Entry 12).
- The meadow definition checked against the EU regulation rather than taken
  from memory.

**Knowledge gaps I still hold:**
- The off-diagonal Hessian terms (Entry 6): I know XGBoost drops them and
  that its built-in multiclass objective does the same, but I have not
  worked through the convergence consequences.
- Whether the orchard/grassland spectral similarity (Entry 13) is actually
  explained by grass cover between trees, or by something else.
- The paired t-tests have 3 degrees of freedom. I treat them as indicative,
  and lean on the 4/4 sign pattern as the more robust evidence.
