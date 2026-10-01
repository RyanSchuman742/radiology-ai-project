"""v2 retrain: one X-ray-pretrained model at 512px, trained on NIH
ChestX-ray14 + VinDr-CXR, with a built-in normal/abnormal output.

Why each change (the deployed models plateaued at ~0.84 macro AUROC on NIH):
- 512px instead of 224px: nodules, thin pneumothorax lines and fine
  fibrotic texture are only a few pixels wide at 224.
- X-ray-pretrained start: TorchXRayVision's densenet121-res224-all instead
  of ImageNet. NOT TXV's resnet50-res512-all, which was trained partly on
  VinDr ("pc-nih-rsna-siim-vin") and so has seen the scoreboard's test images.
  The DenseNet is fully convolutional, so it fine-tunes at 512 fine; its
  classifier rows for our 14 conditions warm-start our head.
- NIH + VinDr: a second hospital, and VinDr's labels are radiologist reads
  rather than NIH's report-mined ones. VinDr is oversampled since its labels
  are the more trustworthy of the two.
- A 15th output, "anything abnormal", trained jointly - replaces the separate
  gate model (one network instead of three at inference).

Partial labels: VinDr labels only 9 of our 14 conditions, so the loss is
masked per (image, output) and only known labels contribute:
- VinDr normal scans (all 3 radiologists: no finding): every output is a
  known negative.
- VinDr abnormal scans: a mapped condition is positive with >=2 of 3 votes,
  negative with 0 votes, unknown with 1 vote. Nodule/Mass is one VinDr class,
  so a positive can't say which of our two it is - both are unknown then.
  Edema, Emphysema, Hernia and Pneumonia aren't labeled by VinDr - unknown.
- NIH: all 14 known (noisy), abnormal = anything other than "No Finding".

Splits: NIH by patient exactly as train_multilabel.py (same test patients);
VinDr from data/vindr/split.csv - its test split is the scoreboard and is
never used here. Model selection uses only the val splits.

Caveat: the TXV starting weights were trained on all of NIH, including our
NIH test patients, so NIH test scores for v2 are optimistic. The VinDr
scoreboard is the trustworthy comparison.

Usage:
    python src/train_v2.py --smoke   # ~100 batches, prints throughput + memory
    python src/train_v2.py           # full run, resumes if interrupted
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchxrayvision as xrv
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from evaluate_vindr import VINDR_TO_OURS
from train_multilabel import CONDITIONS, MODEL_DIR, PROJECT_ROOT, load_manifest, patient_level_split

VINDR_DIR = PROJECT_ROOT / "data" / "vindr"
MODEL_PATH = MODEL_DIR / "xrv_densenet_v2.pt"
LATEST_CHECKPOINT_PATH = MODEL_DIR / "xrv_densenet_v2_latest.pt"
LOG_PATH = MODEL_DIR / "train_v2_history.json"

OUTPUTS = CONDITIONS + ["Abnormal"]
ABNORMAL = len(CONDITIONS)
INIT_WEIGHTS = "densenet121-res224-all"

IMAGE_SIZE = 512
BATCH_SIZE = 16
NUM_EPOCHS = 10
# An "epoch" is a 40k-image sample, not a full pass: DenseNet121 at 512px
# trains at ~18 img/s on the M4 Pro's GPU (compute-bound - data loading
# runs at ~900 img/s, and bf16/fp16 autocast is slower on MPS), so a full
# 101k pass would take ~95 min. 10 x 40k = ~4 passes' worth in ~6.5 h, with
# validation (and early stopping) every ~40 min.
SAMPLES_PER_EPOCH = 40_000
PATIENCE = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 500
VINDR_OVERSAMPLE = 2.0
POS_WEIGHT_DAMPENING = 0.5  # sqrt of neg/pos, same as train_multilabel.py
NUM_WORKERS = 8
SEED = 42


# --- data -------------------------------------------------------------------

def nih_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for path, labels in zip(df["path"], df["labels"]):
        present = set(labels.split("|"))
        target = [float(c in present) for c in CONDITIONS] + [float(labels != "No Finding")]
        rows.append({"path": path, "target": target, "mask": [1.0] * len(OUTPUTS), "source": "nih"})
    return rows


def vindr_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, r in df.iterrows():
        target = [0.0] * len(OUTPUTS)
        mask = [1.0] * len(OUTPUTS)
        if r["abnormal"]:
            mask = [0.0] * len(OUTPUTS)
            target[ABNORMAL] = mask[ABNORMAL] = 1.0
            for vindr_class, ours in VINDR_TO_OURS.items():
                votes = r[vindr_class]
                idx = [CONDITIONS.index(c) for c in ours]
                if votes == 0:
                    for i in idx:
                        mask[i] = 1.0
                elif votes >= 2 and len(idx) == 1:
                    target[idx[0]] = mask[idx[0]] = 1.0
        rows.append({"path": str(VINDR_DIR / "images" / f"{r['image_id']}.png"),
                     "target": target, "mask": mask, "source": "vindr"})
    return rows


def letterbox(image: Image.Image, size: int) -> Image.Image:
    """Fit the longest side to `size` and pad to square, keeping aspect
    ratio (VinDr scans aren't square; squashing would distort the heart)."""
    image = image.copy()
    image.thumbnail((size, size), Image.BILINEAR)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


class XrayDataset(Dataset):
    def __init__(self, rows: list[dict], train: bool):
        self.rows = rows
        # No horizontal flip: it moves the heart to the wrong side.
        self.augment = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=0),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]) if train else None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = letterbox(Image.open(row["path"]).convert("L"), IMAGE_SIZE)
        if self.augment:
            image = self.augment(image)
        pixels = xrv.utils.normalize(np.asarray(image, dtype=np.float32), 255)  # TXV's [-1024, 1024]
        return (torch.from_numpy(pixels)[None], torch.tensor(row["target"]), torch.tensor(row["mask"]))


def load_vindr_split() -> pd.DataFrame:
    split = pd.read_csv(VINDR_DIR / "split.csv")[["image_id", "split"]]
    return split.merge(pd.read_csv(VINDR_DIR / "labels.csv"), on="image_id")


# --- model ------------------------------------------------------------------

class XrayNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = xrv.models.DenseNet(weights=INIT_WEIGHTS if pretrained else None)
        self.features = base.features
        self.head = nn.Linear(base.classifier.in_features, len(OUTPUTS))
        if pretrained:  # warm-start condition rows from TXV's own classifier
            with torch.no_grad():
                for i, c in enumerate(CONDITIONS):
                    j = base.pathologies.index(c)
                    self.head.weight[i] = base.classifier.weight[j]
                    self.head.bias[i] = base.classifier.bias[j]

    def forward(self, x):
        f = F.relu(self.features(x))
        return self.head(F.adaptive_avg_pool2d(f, 1).flatten(1))


def pos_weight(rows: list[dict]) -> torch.Tensor:
    t = torch.tensor([r["target"] for r in rows])
    m = torch.tensor([r["mask"] for r in rows])
    pos = (t * m).sum(0)
    neg = ((1 - t) * m).sum(0)
    return (neg / pos.clamp(min=1)).pow(POS_WEIGHT_DAMPENING)


def masked_bce(logits, target, mask, weight):
    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)


# --- evaluation ---------------------------------------------------------------

def predict(model, loader, device, max_batches=None) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for b, (x, _, _) in enumerate(loader):
            if max_batches and b >= max_batches:
                break
            probs.append(torch.sigmoid(model(x.to(device))).float().cpu().numpy())
    return np.concatenate(probs)


def nih_macro_auroc(probs, rows) -> float:
    t = np.array([r["target"] for r in rows[:len(probs)]])
    aucs = [roc_auc_score(t[:, i], probs[:, i]) for i in range(len(CONDITIONS)) if 0 < t[:, i].sum() < len(t)]
    return float(np.mean(aucs))


def vindr_scores(probs, df) -> dict:
    """Same per-condition rule as the scoreboard: >=2 votes positive,
    0 votes negative, 1 vote left out; Nodule/Mass scored by the max."""
    df = df.iloc[:len(probs)]
    aucs = {}
    for vindr_class, ours in VINDR_TO_OURS.items():
        votes = df[vindr_class].to_numpy()
        keep = (votes >= 2) | (votes == 0)
        if (votes >= 2).sum() >= 30:
            score = probs[:, [CONDITIONS.index(c) for c in ours]].max(axis=1)
            aucs[vindr_class] = float(roc_auc_score(votes[keep] >= 2, score[keep]))
    return {"macro_auroc": float(np.mean(list(aucs.values()))),
            "abnormal_auroc": float(roc_auc_score(df["abnormal"], probs[:, ABNORMAL])),
            "per_condition": aucs}


# --- training -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="time ~100 batches and exit")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dry-run", type=int, metavar="EPOCHS",
                        help="rehearse the full loop with tiny epochs, writing models/dryrun_v2* (delete after)")
    args = parser.parse_args()

    global MODEL_PATH, LATEST_CHECKPOINT_PATH, LOG_PATH, NUM_EPOCHS, SAMPLES_PER_EPOCH
    if args.dry_run:
        MODEL_PATH, LATEST_CHECKPOINT_PATH, LOG_PATH = (MODEL_DIR / f"dryrun_v2{s}" for s in (".pt", "_latest.pt", ".json"))
        NUM_EPOCHS, SAMPLES_PER_EPOCH = args.dry_run, 160

    torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    nih_train, nih_val, _ = patient_level_split(load_manifest())
    vindr = load_vindr_split()
    vindr_val = vindr[vindr["split"] == "val"].reset_index(drop=True)

    train_rows = nih_rows(nih_train) + vindr_rows(vindr[vindr["split"] == "train"])
    if args.dry_run:
        nih_val = nih_val.sample(400, random_state=SEED)
    nih_val_rows, vindr_val_rows = nih_rows(nih_val), vindr_rows(vindr_val)
    print(f"Train: {len(nih_train)} NIH + {(vindr['split'] == 'train').sum()} VinDr | "
          f"val: {len(nih_val)} NIH + {len(vindr_val)} VinDr", flush=True)

    weights = [VINDR_OVERSAMPLE if r["source"] == "vindr" else 1.0 for r in train_rows]
    sampler = WeightedRandomSampler(weights, num_samples=SAMPLES_PER_EPOCH, replacement=True)
    loader_kw = dict(batch_size=args.batch_size, num_workers=NUM_WORKERS, persistent_workers=True)
    train_loader = DataLoader(XrayDataset(train_rows, train=True), sampler=sampler, **loader_kw)
    nih_val_loader = DataLoader(XrayDataset(nih_val_rows, train=False), shuffle=False, **loader_kw)
    vindr_val_loader = DataLoader(XrayDataset(vindr_val_rows, train=False), shuffle=False, **loader_kw)

    model = XrayNet().to(device)
    weight = pos_weight(train_rows).to(device)
    print("Pos weights: " + ", ".join(f"{o}={w:.1f}" for o, w in zip(OUTPUTS, weight.tolist())), flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = NUM_EPOCHS * len(train_loader)
    # float(): a numpy scalar here ends up in the checkpoint, which torch.load's safe mode refuses on resume
    schedule = lambda step: float(min(1.0, (step + 1) / WARMUP_STEPS) * 0.5 * (1 + np.cos(np.pi * min(step, total_steps) / total_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    if args.smoke:
        smoke(model, train_loader, nih_val_loader, vindr_val_loader, vindr_val, nih_val_rows,
              optimizer, scheduler, weight, device)
        return

    history, best, start_epoch, stale = [], -1.0, 1, 0
    if LATEST_CHECKPOINT_PATH.exists():  # resume an interrupted overnight run
        ckpt = torch.load(LATEST_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        history, best, stale = ckpt["history"], ckpt["best"], ckpt["stale"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch} (best selection score {best:.4f})", flush=True)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        t0 = time.time()
        model.train()
        running, n = 0.0, 0
        for step, (x, target, mask) in enumerate(train_loader):
            x, target, mask = x.to(device), target.to(device), mask.to(device)
            loss = masked_bce(model(x), target, mask, weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running, n = running + loss.item(), n + 1
            if step % 500 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss={running / n:.4f} "
                      f"({(time.time() - t0) / 60:.0f} min)", flush=True)

        nih_auc = nih_macro_auroc(predict(model, nih_val_loader, device), nih_val_rows)
        vin = vindr_scores(predict(model, vindr_val_loader, device), vindr_val)
        selection = float(np.mean([nih_auc, vin["macro_auroc"], vin["abnormal_auroc"]]))
        entry = {"epoch": epoch, "train_loss": running / n, "nih_val_macro_auroc": nih_auc,
                 "vindr_val_macro_auroc": vin["macro_auroc"], "vindr_val_abnormal_auroc": vin["abnormal_auroc"],
                 "vindr_val_per_condition": vin["per_condition"], "selection": selection,
                 "minutes": (time.time() - t0) / 60}
        history.append(entry)
        print(f"Epoch {epoch}/{NUM_EPOCHS} | loss={entry['train_loss']:.4f} | NIH val macro={nih_auc:.4f} | "
              f"VinDr val macro={vin['macro_auroc']:.4f} abnormal={vin['abnormal_auroc']:.4f} | "
              f"selection={selection:.4f} | {entry['minutes']:.0f} min", flush=True)

        if selection > best:
            best, stale = selection, 0
            torch.save({"model_state": model.state_dict(), "outputs": OUTPUTS, "init_weights": INIT_WEIGHTS,
                        "image_size": IMAGE_SIZE, "epoch": epoch, "selection": selection}, MODEL_PATH)
            print(f"  Saved new best model to {MODEL_PATH}", flush=True)
        else:
            stale += 1

        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(), "epoch": epoch, "best": best,
                    "stale": stale, "history": history}, LATEST_CHECKPOINT_PATH)
        LOG_PATH.write_text(json.dumps(history, indent=2))

        if stale >= PATIENCE:
            print(f"No improvement for {PATIENCE} epochs - stopping early.", flush=True)
            break

    print(f"Done. Best selection score {best:.4f}; model at {MODEL_PATH}", flush=True)


def smoke(model, train_loader, nih_val_loader, vindr_val_loader, vindr_val, nih_val_rows,
          optimizer, scheduler, weight, device):
    model.train()
    t0, n_batches = None, 100
    for step, (x, target, mask) in enumerate(train_loader):
        if step == 5:  # skip warm-up batches (worker spin-up, MPS kernel compilation)
            t0 = time.time()
        x, target, mask = x.to(device), target.to(device), mask.to(device)
        loss = masked_bce(model(x), target, mask, weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 20 == 0:
            print(f"  step {step} loss={loss.item():.4f}", flush=True)
        if step == n_batches + 5:
            break
    torch.mps.synchronize()
    train_rate = n_batches * train_loader.batch_size / (time.time() - t0)

    t1 = time.time()
    probs = predict(model, vindr_val_loader, device, max_batches=20)
    eval_rate = len(probs) / (time.time() - t1)

    n_train = SAMPLES_PER_EPOCH
    n_val = len(nih_val_loader.dataset) + len(vindr_val_loader.dataset)
    epoch_min = (n_train / train_rate + n_val / eval_rate) / 60
    print(f"\nTrain {train_rate:.1f} img/s, eval {eval_rate:.1f} img/s")
    print(f"Estimated epoch: {epoch_min:.0f} min -> {NUM_EPOCHS} epochs = {NUM_EPOCHS * epoch_min / 60:.1f} h")
    print(f"MPS memory: {torch.mps.driver_allocated_memory() / 1e9:.1f} GB allocated by driver")
    print(f"Sanity - VinDr val on {len(probs)} images: {vindr_scores(probs, vindr_val)}")


if __name__ == "__main__":
    main()
