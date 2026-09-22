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

**Pneumonia and Infiltration are the weakest classes in both this run and an
earlier subset-only run** - consistent enough across two independent training
runs that this looks like a genuinely harder class for this architecture
(possibly due to diffuse, poorly-bounded visual presentation), not sampling
noise.

## Known limitations

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
