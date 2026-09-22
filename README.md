---
title: Chest X-ray Interpretability Tool
emoji: 🫁
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Chest X-ray Interpretability Tool

Upload a chest X-ray, get back predicted findings across 14 conditions, a
color-coded Grad-CAM heatmap showing what the model looked at for each one,
and a plain-English explanation you can use to sanity-check the reasoning.

**Educational prototype — not a medical device, not a diagnosis, not a
substitute for a radiologist.** See [NIH14_MODEL_CARD.md](NIH14_MODEL_CARD.md)
(current, multi-label model) or [MODEL_CARD.md](MODEL_CARD.md) (earlier
binary pneumonia/normal model) for training data, measured performance, and
known limitations.

## Stack

- PyTorch (ResNet50, fine-tuned, MPS/CPU), multi-label (`BCEWithLogitsLoss`),
  ensembled with [TorchXRayVision](https://github.com/mlmed/torchxrayvision)'s
  multi-dataset DenseNet121 for predictions
- `pytorch-grad-cam` for per-condition heatmaps, composited into one
  color-coded image
- Flask for the web app
- Anthropic API (Claude, vision) for the explanation layer - reads the
  actual heatmap image, not just the numbers

## Local setup

```bash
conda activate radiology-ai
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

Train the classifier (downloads the full NIH-14 dataset first if you
haven't - ~42GB, run overnight):

```bash
python src/download_nih14.py
python src/train_multilabel.py
```

Run the app:

```bash
python app.py
```

## Deploying (Google Cloud Run)

```bash
gcloud run deploy radiology-ai-project \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-secrets=ANTHROPIC_API_KEY=anthropic-api-key:latest \
  --memory 2Gi --cpu 2 --min-instances 0 --max-instances 3 --timeout 300
```

The trained model checkpoint (`models/resnet_nih14.pt`) is committed to the
repo so no training step is needed at deploy time. `ANTHROPIC_API_KEY` is
stored in Secret Manager, never in source.

## Data attribution

Current model trained on [NIH ChestX-ray14](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset)
(Wang et al., 2017), public domain, NIH Clinical Center. Not included in
this repo — run `src/download_nih14.py` to fetch it (~42GB).

Earlier binary model trained on the Kermany et al. (2018) pediatric
pneumonia dataset (CC-BY-4.0) - see `src/download_pneumonia_dataset.py`.

## Roadmap

- **Done:** binary (Normal vs. Pneumonia) → multi-label, 14 conditions
  (NIH ChestX-ray14), per-condition color-coded Grad-CAM heatmaps
- Researched, not yet built: calibrated uncertainty (conformal prediction)
  instead of raw sigmoid confidence; CXR Foundation embeddings instead of
  ImageNet pretraining; prior-scan comparison; symptom/clinical-context
  input alongside the image
- Longer term: additional imaging modalities beyond chest X-ray
