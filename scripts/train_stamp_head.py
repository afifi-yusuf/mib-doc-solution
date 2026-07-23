#!/usr/bin/env python3
"""Train the tiny stamp/risk raster head from public train labels.

Positive: packets whose gold risk_flags include a hard flag.
Negative: gold APPROVED with risk_flags=none (clean packets).
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import fitz
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "mib_solution" / "assets" / "stamp_head.pt"
HARD = {
    "biohazard_red",
    "active_warrant",
    "memory_tampering",
    "planetary_embargo",
    "illegible_biometrics",
}


def iter_rasters(pdf: Path, min_side: int = 200):
    doc = fitz.open(pdf)
    try:
        for page in doc:
            for img in page.get_images(full=True):
                try:
                    pix = fitz.Pixmap(doc, img[0])
                except Exception:
                    continue
                if pix.n < 3 or min(pix.width, pix.height) < min_side:
                    continue
                yield np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )[:, :, :3]
    finally:
        doc.close()


def preprocess(arr: np.ndarray, size: int = 64) -> np.ndarray:
    im = Image.fromarray(arr).convert("RGB").resize((size, size))
    x = np.asarray(im, dtype=np.float32) / 255.0
    return np.transpose(x, (2, 0, 1))


class RasterDataset(Dataset):
    def __init__(self, items: list[tuple[np.ndarray, int]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        arr, label = self.items[index]
        return torch.from_numpy(preprocess(arr)), torch.tensor(label, dtype=torch.float32)


class TinyStampNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main() -> None:
    labels = list(csv.DictReader((ROOT / "data" / "train_labels.csv").open()))
    items: list[tuple[np.ndarray, int]] = []
    pos_packets = 0
    neg_packets = 0
    for row in labels:
        flags = set(str(row["risk_flags"]).split("|")) - {"", "none"}
        hard = bool(flags & HARD)
        if hard:
            label = 1
        elif row["adjudication"] == "APPROVED" and row["risk_flags"] == "none":
            label = 0
        else:
            continue
        pdf = ROOT / "data" / "train" / f"{row['case_id']}.pdf"
        rasters = list(iter_rasters(pdf))
        if not rasters:
            continue
        if label == 1:
            pos_packets += 1
        else:
            neg_packets += 1
        # Keep up to 3 rasters per packet to limit imbalance.
        random.shuffle(rasters)
        for arr in rasters[:3]:
            items.append((arr, label))
    print(f"packets pos={pos_packets} neg={neg_packets} rasters={len(items)}")
    random.shuffle(items)
    # Balance roughly
    pos = [it for it in items if it[1] == 1]
    neg = [it for it in items if it[1] == 0]
    n = min(len(pos) * 3, len(neg))
    balanced = pos + neg[:n]
    random.shuffle(balanced)
    print(f"balanced={len(balanced)} pos={sum(1 for _,y in balanced if y==1)}")

    ds = RasterDataset(balanced)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    model = TinyStampNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(8):
        total = 0.0
        n = 0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yb)
            n += len(yb)
        print(f"epoch {epoch+1} loss={total/max(n,1):.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT)
    print(f"wrote {OUT} size={OUT.stat().st_size}")


if __name__ == "__main__":
    main()
