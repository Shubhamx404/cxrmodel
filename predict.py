import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(1)

DEVICE = torch.device("cpu")

# -----------------------------
# Transforms (MATCHES NOTEBOOK)
# -----------------------------
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# -----------------------------
# Model builder (MATCHES NOTEBOOK)
# -----------------------------
def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)
    return model

# -----------------------------
# Safe probability normalization
# -----------------------------
def normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 0.0, 1.0)
    s = probs.sum()
    if s > 0:
        return probs / s
    return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

# -----------------------------
# Main inference
# -----------------------------
def main(args):
    images_dir = Path(args.images_dir)
    metadata_csv = Path(args.metadata_csv)
    checkpoint_path = Path(args.checkpoint)
    out_csv = Path(args.out_csv)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    assert images_dir.exists(), f"Images dir not found: {images_dir}"
    assert metadata_csv.exists(), f"Metadata CSV not found: {metadata_csv}"
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"

    # Load metadata
    meta = pd.read_csv(metadata_csv)
    assert "image_id" in meta.columns, "metadata.csv must contain image_id column"

    image_ids = meta["image_id"].tolist()

    # Load model
    model = build_model().to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    rows = []

    with torch.no_grad():
        for image_id in tqdm(image_ids, desc="Running inference"):
            img_path = images_dir / image_id
            if not img_path.exists():
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            x = VAL_TRANSFORM(img).unsqueeze(0).to(DEVICE)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            probs = normalize_probs(probs)

            rows.append({
                "image_id": image_id,
                "p0": float(probs[0]),
                "p1": float(probs[1]),
                "p2": float(probs[2]),
                "p3": float(probs[3]),
            })

    df = pd.DataFrame(rows, columns=["image_id", "p0", "p1", "p2", "p3"])
    df.to_csv(out_csv, index=False)

    print(f"✅ Predictions written to: {out_csv}")
    print(f"Total predictions: {len(df)}")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ImagingIQ CXR Triage Inference")

    parser.add_argument("--images_dir", required=True, type=str)
    parser.add_argument("--metadata_csv", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--out_csv", required=True, type=str)

    args = parser.parse_args()
    main(args)
