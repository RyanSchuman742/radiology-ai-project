# Chest X-ray Interpretability Tool

Upload a chest X-ray, get back a predicted diagnosis, a Grad-CAM heatmap
showing what the model looked at, and a plain-English explanation you can
use to sanity-check the reasoning.

**Educational prototype — not a medical device, not a diagnosis, not a
substitute for a radiologist.**

## Stack

- PyTorch (ResNet50, fine-tuned, MPS/CPU) for classification
- `pytorch-grad-cam` for the heatmap overlay
- Flask for the web app
- Anthropic API (Claude) for the explanation layer

## Local setup

```bash
conda activate radiology-ai
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

Train the classifier (downloads the dataset first if you haven't):

```bash
python src/download_pneumonia_dataset.py
python src/train_classifier.py
```

Run the app:

```bash
python app.py
```

## Deploying (Railway)

1. Push this repo to GitHub.
2. In Railway, create a new project from that GitHub repo.
3. Set the `ANTHROPIC_API_KEY` environment variable in Railway's dashboard.
4. Railway will detect the `Procfile` and run `gunicorn app:app`.

The trained model checkpoint (`models/resnet_pneumonia.pt`) is committed to
the repo so no training step is needed at deploy time.

## Data attribution

Training data: Kermany, D.; Zhang, K.; Goldbaum, M. (2018), "Labeled Optical
Coherence Tomography (OCT) and Chest X-Ray Images for Classification",
mirrored at
[hf-vision/chest-xray-pneumonia](https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia)
(CC-BY-4.0). Not included in this repo — run
`src/download_pneumonia_dataset.py` to fetch it.

## Roadmap

- Currently binary: Normal vs. Pneumonia. Plan to expand to more conditions
  (e.g. via the NIH ChestX-ray14 dataset) once the pipeline is validated.
