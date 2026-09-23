# Model Card — Multi-Label Chest X-ray Classifier (NIH ChestX-ray14)

**Educational prototype. Not a medical device, not a diagnosis, not FDA-cleared.
Do not use for real clinical decisions.**

## What it does

Multi-label classification of a frontal chest X-ray across 14 conditions
simultaneously (a scan can have zero, one, or several findings at once),
with a per-condition Grad-CAM heatmap composited into one color-coded image
so multiple co-occurring findings each show their own region.

Conditions: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion,
Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural_Thickening,
Pneumonia, Pneumothorax. ("No Finding" is implicit - a scan with none of the
14 above threshold.)

## Architecture

- ResNet50, pretrained on ImageNet, fine-tuned end-to-end
- Input: 224x224, converted to RGB, ImageNet normalization
- Trained on Apple Silicon (MPS backend), Adam optimizer, lr=1e-4
- Loss: `BCEWithLogitsLoss` with per-class `pos_weight` (independent sigmoid
  per class, not softmax - multiple findings can be true at once)
- **Deployed predictions are an ensemble** - the average of this ResNet50 and
  TorchXRayVision's DenseNet121, with per-class thresholds calibrated for the
  averaged output. See "Ensemble with TorchXRayVision" below. Grad-CAM
  heatmaps come from this ResNet50 only.
- **A separate normal/abnormal gate** decides how findings are *presented*:
  when it rates a scan likely normal, the page leads with "No significant
  abnormality detected" and moves any flagged findings into a collapsed
  low-confidence section. Findings are never hidden. See "Normal/abnormal
  gate" below.

## Training data

[NIH ChestX-ray14](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset)
(Wang et al., 2017), the full official release, public domain, NIH Clinical
Center. 112,120 images from 30,805 unique patients.

**Split by patient, not by image** - NIH-14 has multiple follow-up X-rays per
patient, and splitting by image would let the same patient's anatomy leak
across train/val/test, inflating reported metrics. This is a known pitfall
in published work on this dataset.

| Split | Patients | Images |
|---|---|---|
| Train | 24,645 | 90,940 |
| Validation | 3,080 | 10,649 |
| Test (held out, never trained on) | 3,080 | 10,531 |

Labels were NLP-mined from the original radiology reports by the dataset's
authors, not verified by a radiologist for this release - some label noise
is expected and documented in the original paper.

## Class imbalance

Severe and uneven across 14 classes - from Infiltration (17.7% of images) to
Hernia (0.2%). Per-class `pos_weight` in the loss function corrects for this
(computed from the training split; Hernia's weight was ~478x, the highest).

## Training run

20 epochs, ~4 hours unattended. Validation macro AUROC peaked at **epoch 11**
(0.834); later epochs overfit (train AUROC kept climbing to 0.93 while
validation plateaued and validation loss worsened). The epoch-11 checkpoint
was kept as the deployed model, not the final epoch.

## Measured performance (held-out test set, 10,531 images / 3,080 patients)

**Macro AUROC: 0.826** (unweighted average across all 14 conditions).

| Condition | AUROC | Test support |
|---|---|---|
| Emphysema | 0.931 | 195 |
| Cardiomegaly | 0.917 | 250 |
| Edema | 0.898 | 170 |
| Hernia | 0.897 | 15 |
| Pneumothorax | 0.864 | 447 |
| Effusion | 0.870 | 1,091 |
| Mass | 0.820 | 451 |
| Consolidation | 0.819 | 427 |
| Atelectasis | 0.807 | 1,163 |
| Pleural_Thickening | 0.798 | 297 |
| Fibrosis | 0.768 | 150 |
| Nodule | 0.761 | 612 |
| Infiltration | 0.707 | 1,808 |
| Pneumonia | 0.705 | 117 |

For context, the original CheXNet paper (Rajpurkar et al., 2017) reported a
mean AUROC of ~0.841 on this same dataset. Full per-class precision/recall/F1
is in `models/nih14_eval_report.json` (regenerate with `src/train_multilabel.py`).

## Decision threshold calibration

AUROC measures ranking quality, not what happens at the actual decision
threshold used to show a finding. With a flat 0.5 threshold, **62.3% of
genuinely healthy (No Finding) test images got at least one false-positive
finding flagged** - a real problem reported after trying the deployed app
on outside images, not just a dataset quirk. Root cause: `pos_weight` in
the loss function (needed to handle the severe class imbalance - Hernia's
weight was ~478x) inflates predicted probabilities across the board, so 0.5
stopped meaning the same thing for every class.

`src/calibrate_thresholds.py` finds a per-class threshold on the validation
set (never the test set) that only ever raises the threshold above 0.5, and
only where doing so keeps recall within 90% of its value at 0.5 - i.e. it
only takes precision gains that don't meaningfully cost sensitivity. Two
more aggressive approaches were tried and rejected first: maximizing F1
let some classes' recall collapse below 10% (Pneumonia: 50%->6%) chasing
precision; a flat 75%-recall floor forced already-hard-to-separate classes
*below* 0.5, making the false-positive rate worse (62%->80%), not better.

**Result: 53.3% of healthy test images still get flagged (down from 62.3%),
and every class's precision improved with only a modest recall cost
(typically 4-9 points).** This is a real but limited fix. The residual
53% false-positive rate reflects a genuine limitation of the current
model's probability separation for several classes - training-level fixes
(temperature scaling, revisiting how aggressively `pos_weight` corrects
for imbalance, more regularization) would likely help further; post-hoc
thresholding alone can't fully solve it.

| Condition | Threshold | Recall (0.5 -> calibrated) | Precision (0.5 -> calibrated) |
|---|---|---|---|
| Hernia | 0.850 | 0.600 -> 0.600 | 0.093 -> 0.188 |
| Emphysema | 0.693 | 0.815 -> 0.774 | 0.124 -> 0.182 |
| Cardiomegaly | 0.640 | 0.764 -> 0.684 | 0.154 -> 0.198 |
| Effusion | 0.621 | 0.778 -> 0.688 | 0.311 -> 0.366 |
| Edema | 0.612 | 0.782 -> 0.729 | 0.089 -> 0.106 |
| Pneumothorax | 0.592 | 0.745 -> 0.700 | 0.176 -> 0.204 |
| Atelectasis | 0.581 | 0.810 -> 0.730 | 0.227 -> 0.255 |
| Mass | 0.597 | 0.643 -> 0.574 | 0.166 -> 0.190 |
| Consolidation | 0.571 | 0.721 -> 0.644 | 0.126 -> 0.141 |
| Pleural_Thickening | 0.565 | 0.653 -> 0.623 | 0.088 -> 0.101 |
| Pneumonia | 0.567 | 0.504 -> 0.462 | 0.032 -> 0.038 |
| Fibrosis | 0.555 | 0.613 -> 0.573 | 0.042 -> 0.045 |
| Nodule | 0.550 | 0.585 -> 0.518 | 0.159 -> 0.176 |
| Infiltration | 0.537 | 0.518 -> 0.468 | 0.322 -> 0.352 |

**Pneumonia and Infiltration are the weakest classes in both this run and an
earlier subset-only run** - consistent enough across two independent training
runs that this looks like a genuinely harder class for this architecture
(possibly due to diffuse, poorly-bounded visual presentation), not sampling
noise.

## Tried and rejected: temperature scaling

After threshold calibration still left a 53.3% false-positive rate on
healthy images, tried per-class temperature scaling (`src/temperature_scale.py`,
Guo et al. 2017) - rescaling each class's logit by a learned constant
before the sigmoid, to fix the probability *values* rather than the
decision threshold. **Result: no improvement.** Healthy-image false-positive
rate was unchanged (53.3% -> 53.3%), and Expected Calibration Error got
*worse* for most classes (e.g. Atelectasis 0.318->0.332, Pneumothorax
0.192->0.208). Several classes' learned temperatures came out below 1
(Nodule 0.821, Infiltration 0.860), meaning those classes were actually
*under*confident on validation data, not overconfident - contradicting the
assumption that `pos_weight` uniformly inflates every class's probabilities.
This means the miscalibration is more complex than a single per-class
scalar can correct; full report in `models/nih14_temperature_report.json`.
Not deployed - the model in production is the threshold-only-calibrated
version, unaffected by this experiment.

## Ensemble with TorchXRayVision (deployed)

Averages this model's probabilities with those of TorchXRayVision's
`densenet121-res224-all` (Cohen et al.), a published model trained on
several combined public chest X-ray datasets (NIH, PadChest, CheXpert,
MIMIC-CXR, and others). Its 18 outputs include all 14 of ours under
identical names; the other 4 are ignored. Evaluation: `src/evaluate_ensemble.py`,
full numbers in `models/nih14_ensemble_report.json`.

**Contamination caveat:** TXV's weights were trained partly on NIH
ChestX-ray14, so it has likely seen some of our "held-out" NIH test images.
The NIH test numbers below are therefore optimistic for TXV and the
ensemble. The **Kermany pediatric NORMAL set** (1,583 healthy X-rays from a
different hospital, trained on by neither model) is the clean check - and
the closest match to the original bug report, which was healthy X-rays from
outside the training data being flagged.

| | Ours alone (previous) | TXV alone | Ensemble (deployed) |
|---|---|---|---|
| Healthy *external* X-rays with >=1 false positive | 79.3% | 94.8% | **53.1%** |
| Healthy NIH test X-rays with >=1 false positive | 53.3% | 86.9% | **46.6%** |
| NIH test macro AUROC | 0.826 | 0.775 | **0.834** |

Each model alone is poor on outside images, but they make different
mistakes, so averaging cancels many of them out. The external number also
revealed that our model alone was much worse on outside images (79%) than
the 53% measured on NIH data suggested.

**Thresholds match our model's existing sensitivity.** A first version used
the same conservative search as `calibrate_thresholds.py` and cut false
positives slightly more (50.7% external), but dropped Pneumonia recall from
46% to 33%. Since the ensemble *ranks* Pneumonia cases better than our
model does (AUROC 0.733 vs 0.705), that drop was a threshold choice, not a
weaker model - so each class's threshold is instead set, on the validation
set, to catch at least as many real cases as our model alone did. Test-set
recall ends up within about 1-4 points of the previous model for every
class except:

- **Hernia: 60% -> 40%** (9 -> 6 of 15 test cases). With 15 cases this is
  mostly noise, and tuning against the test set to "fix" it would be
  cheating. Unresolved.
- Emphysema: 77% -> 74%.

## Normal/abnormal gate (deployed, demotes rather than hides)

**Why:** the app runs 14 separate per-condition tests, and a healthy scan
counts as a false alarm if *any* of them fires. Even at 5% false alarms
per condition, the chance at least one of 14 fires is about 51% - roughly
what we measured. Per-condition tuning can't fix that structure. A single
model answering "is anything abnormal here?" can.

**Model:** `src/train_gate.py` - ResNet50 (ImageNet init, 224px, same
recipe and patient split as the 14-condition model), trained on NIH "No
Finding" vs. any finding, 12 epochs, best at epoch 8. **Test AUROC 0.772**
- lower than hoped, likely because NIH's "No Finding" labels are
report-mined and noisy. On its own it clears only 26% of healthy scans at
95% sensitivity, 39% at 90%.

**End-to-end** (`src/evaluate_gated_system.py`, gate threshold chosen on
validation at 90% sensitivity):

| | Ensemble, no gate | With gate as a hard veto |
|---|---|---|
| Healthy *external* X-rays flagged | 53.1% | **35.2%** |
| Healthy NIH X-rays flagged | 46.6% | **40.6%** |
| External *pneumonia* X-rays still flagged | 91.2% | 89.2% |

As a hard veto it would cost recall on a few conditions (Cardiomegaly
68%->63%, Fibrosis 56%->52%, Nodule 53%->51%; others within 1 point). So
the deployed app **demotes instead of vetoing**: a gate-cleared scan leads
with "No significant abnormality detected", but every flagged finding is
still shown in a collapsed low-confidence section, and the explanation
layer is told to frame them as low-confidence. The headline gets the
false-alarm improvement above; nothing a clinician could review is lost.

A stricter gate (95% sensitivity) barely helps (46.4% external), and 97.5%
or 99% does nearly nothing. The gate's ceiling is most likely label quality:
retraining it on radiologist-verified "No finding" labels (VinDr-CXR) is
the planned next step.

## Known limitations

- **Still flags a large share of genuinely healthy X-rays** - with the
  gate, 35.2% of external healthy images and 40.6% of NIH healthy images
  still lead with a finding (down from 79.3% / 53.3% for the original single
  model). A known, unresolved limitation, not a hidden one.
- NIH labels are report-mined, so some measured "false positives" on NIH
  are likely real findings the labels missed - a radiologist-verified test
  set (VinDr-CXR) is needed to know the true rate.
- The external healthy check is pediatric (Kermany), while training data is
  mostly adult - some of the external false-positive rate may be age-related
  domain shift rather than general over-flagging.
- Label noise from NLP-mined (not radiologist-verified) ground truth
- Single-institution data source, no external validation set
- Overfitting past epoch ~11 - the model would likely benefit from
  regularization, more aggressive augmentation, or fewer epochs, not just
  more data
- Grad-CAM shows correlation with the prediction, not a guarantee the
  highlighted region is clinically meaningful
- Composite multi-color heatmap: when two conditions' attention overlaps in
  the same anatomical region, their colors blend and become visually
  ambiguous - individual per-condition heatmaps are provided specifically
  to resolve this, but the composite alone can be misleading in isolation
- No comparison against a prior scan for the same patient (single-timepoint
  only)
- No symptom/clinical-context input - the model (and its Grad-CAM) only ever
  sees the image, unlike a radiologist who also has the clinical indication

## Intended use

Coursework / portfolio demonstration of an interpretable, multi-label ML
pipeline. Not intended for, and not validated for, real diagnostic use.
