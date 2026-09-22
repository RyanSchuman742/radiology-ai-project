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
- Decision threshold: 0.5 per class

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

## Known limitations

- **Still flags over half of genuinely healthy images with at least one
  false positive (53.3%), even after threshold calibration.** See
  "Decision threshold calibration" above - this is a known, unresolved
  limitation, not a hidden one.
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
