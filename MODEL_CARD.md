# Model Card — Chest X-ray Pneumonia Classifier

**Educational prototype. Not a medical device, not a diagnosis, not FDA-cleared.
Do not use for real clinical decisions.**

## What it does

Binary classification of a frontal chest X-ray as **Normal** or **Pneumonia**,
with a Grad-CAM heatmap showing the image regions that drove the prediction.

## Architecture

- ResNet50, pretrained on ImageNet, fine-tuned end-to-end
- Input: 224x224, converted to RGB, ImageNet normalization
- Trained on Apple Silicon (MPS backend), Adam optimizer, lr=1e-4, 5 epochs
- Loss: class-weighted cross-entropy (see "Class imbalance" below)

## Training data

[Kermany et al. (2018)](https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia)
pediatric chest X-ray dataset, CC-BY-4.0. 5,856 images total:

| Split | Normal | Pneumonia | Total |
|---|---|---|---|
| Train | 1,191 | 3,504 | 4,695 |
| Validation | 150 | 387 | 537 |
| Test (held out, never trained on) | 234 | 390 | 624 |

**This dataset is pediatric only** (children aged 1-5) and sourced from a
single medical center in Guangzhou, China. A model trained on it should not
be assumed to generalize to adult patients, other patient populations, or
X-rays from different equipment/imaging protocols.

## Class imbalance

The raw training set is ~74% pneumonia / 26% normal. An earlier version of
this model, trained with unweighted cross-entropy, learned to over-predict
the majority class. The loss function now weights each class inversely to
its frequency (Normal: 3.94x, Pneumonia: 1.34x) to correct for this.

## Measured performance (held-out test set, 624 images)

| Metric | Unweighted (before fix) | Class-weighted (current) |
|---|---|---|
| Accuracy | 81.9% | **90.4%** |
| AUROC | 0.963 | 0.968 |
| Sensitivity (recall on Pneumonia) | 99.7% | 98.5% |
| Specificity (recall on Normal) | 52.1% | **76.9%** |
| False negatives (missed pneumonia) | 1 / 390 | 6 / 390 |
| False positives (false alarms) | 112 / 234 | 54 / 234 |

Sensitivity is the number that matters most for a screening tool — missing
a real case is worse than a false alarm — and it stayed high (98.5%) even
after fixing the false-alarm problem. Full numbers, including per-class
precision/recall/F1, are in `models/eval_report.json` (regenerate with
`src/evaluate.py`).

**Specificity at 76.9% is still not great.** Roughly 1 in 4 normal X-rays
gets flagged as pneumonia. This is a known limitation, not a hidden one.

## Known limitations

- Binary only — cannot detect any condition other than pneumonia
- Pediatric training data only
- Single source hospital — no diversity in equipment/protocol
- No external validation set from a different institution
- Grad-CAM shows correlation with the prediction, not a guarantee that the
  highlighted region is clinically meaningful
- The explanation layer (Claude) never sees the image — it only reasons
  over the numbers this model already produced, so it cannot catch a case
  where the model's number is wrong

## Intended use

Coursework / portfolio demonstration of an interpretable ML pipeline. Not
intended for, and not validated for, real diagnostic use.
