"""Weakly supervised local-stamp CNN experiment.

The proposal stage is fully deterministic: it selects the two grid cells with
the strongest red/blue ink density on each rendered page.  Training receives
only packet-level risk labels; no stamp boxes or validation labels are used.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

FLAGS = ("memory_tampering", "active_warrant", "biohazard_red", "planetary_embargo")
PATCH_SIZE = 96
PATCHES_PER_PAGE = 2
MAX_PAGES = 6


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def proposed_patches(path: Path) -> torch.Tensor:
    """Return colour-ink proposals without inspecting labels or text."""
    original = Image.open(path).convert("RGB")
    small = np.asarray(original.resize((160, 208)), dtype=np.float32) / 255.0
    red = (small[..., 0] > small[..., 1] * 1.22) & (small[..., 0] > small[..., 2] * 1.22) & (small[..., 0] > .32)
    blue = (small[..., 2] > small[..., 0] * 1.12) & (small[..., 2] > small[..., 1] * 1.04) & (small[..., 2] > .28)
    score = red.astype(np.float32) + blue.astype(np.float32)
    height, width = score.shape
    candidates: list[tuple[float, int, int]] = []
    for row in range(4):
        for col in range(4):
            y0, y1 = row * height // 4, (row + 1) * height // 4
            x0, x1 = col * width // 4, (col + 1) * width // 4
            candidates.append((float(score[y0:y1, x0:x1].mean()), row, col))
    candidates.sort(reverse=True)
    output: list[torch.Tensor] = []
    for _, row, col in candidates[:PATCHES_PER_PAGE]:
        # A padded cell keeps complete tilted/offset stamps in view.
        x0, x1 = (col - .25) * original.width / 4, (col + 1.25) * original.width / 4
        y0, y1 = (row - .25) * original.height / 4, (row + 1.25) * original.height / 4
        crop = original.crop((max(0, int(x0)), max(0, int(y0)), min(original.width, int(x1)), min(original.height, int(y1))))
        pixels = torch.from_numpy(np.asarray(crop.resize((PATCH_SIZE, PATCH_SIZE))).copy()).to(torch.float32)
        output.append(pixels.permute(2, 0, 1).div(255.0))
    return torch.stack(output)


class Packets(Dataset):
    def __init__(self, rows: list[dict[str, str]], cache: Path, augment: bool = False):
        self.rows, self.cache, self.augment = rows, cache, augment
        self.memo: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        value = self.memo.get(row["case_id"])
        if value is None:
            pages = sorted((self.cache / row["case_id"]).glob("page-*.png"))[:MAX_PAGES]
            value = torch.cat([proposed_patches(page) for page in pages])
            if len(pages) < MAX_PAGES:
                value = torch.cat([value, torch.zeros((MAX_PAGES - len(pages)) * PATCHES_PER_PAGE, 3, PATCH_SIZE, PATCH_SIZE)])
            self.memo[row["case_id"]] = value
        mask = value.abs().sum((1, 2, 3)) > 0
        if self.augment:
            value = (value * float(torch.empty(1).uniform_(.88, 1.14))).clamp(0, 1)
        risk = set(row["risk_flags"].split("|"))
        return value, mask, torch.tensor([flag in risk for flag in FLAGS], dtype=torch.float32), row["case_id"]


class LocalStampMIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2, 2), nn.BatchNorm2d(24), nn.SiLU(),
            nn.Conv2d(24, 48, 3, 2, 1), nn.BatchNorm2d(48), nn.SiLU(),
            nn.Conv2d(48, 72, 3, 2, 1), nn.BatchNorm2d(72), nn.SiLU(),
            nn.Conv2d(72, 96, 3, 2, 1), nn.BatchNorm2d(96), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Dropout(.2), nn.Linear(96, len(FLAGS)))

    def forward(self, patches: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, count, channels, height, width = patches.shape
        logits = self.head(self.encoder(patches.reshape(batch * count, channels, height, width)).flatten(1))
        return logits.reshape(batch, count, len(FLAGS)).masked_fill(~mask.unsqueeze(-1), -30).amax(1)


def train(model: LocalStampMIL, loader: DataLoader, target: torch.device, epochs: int, weights: torch.Tensor) -> None:
    model.to(target)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights.to(target))
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
    for epoch in range(epochs):
        model.train()
        losses = []
        for patches, mask, labels, _ in loader:
            loss = loss_fn(model(patches.to(target), mask.to(target)), labels.to(target))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"patch-stamp epoch {epoch + 1}/{epochs}: loss={sum(losses) / len(losses):.4f}", flush=True)


@torch.no_grad()
def predict(model: LocalStampMIL, loader: DataLoader, target: torch.device) -> dict[str, dict[str, float]]:
    model.eval()
    output = {}
    for patches, mask, _, case_ids in loader:
        values = torch.sigmoid(model(patches.to(target), mask.to(target))).cpu()
        for case_id, row in zip(case_ids, values):
            output[case_id] = {flag: float(value) for flag, value in zip(FLAGS, row)}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--oof", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.labels.open()))
    target = device()
    print(f"Patch-stamp CNN device: {target}", flush=True)
    risk_any = ["1" if any(flag in row["risk_flags"].split("|") for flag in FLAGS) else "0" for row in rows]
    positive = torch.tensor([sum(flag in row["risk_flags"].split("|") for row in rows) for flag in FLAGS], dtype=torch.float32)
    weights = (len(rows) - positive).clamp_min(1) / positive.clamp_min(1)
    output: dict[str, dict[str, float]] = {}
    folds = StratifiedKFold(args.folds, shuffle=True, random_state=8090)
    for number, (train_idx, valid_idx) in enumerate(folds.split(rows, risk_any), start=1):
        print(f"Patch-stamp OOF fold {number}/{args.folds}", flush=True)
        model = LocalStampMIL()
        train_rows = [rows[index] for index in train_idx]
        valid_rows = [rows[index] for index in valid_idx]
        train(model, DataLoader(Packets(train_rows, args.cache, True), batch_size=12, shuffle=True), target, args.epochs, weights)
        output.update(predict(model, DataLoader(Packets(valid_rows, args.cache), batch_size=12), target))
    with args.oof.open("w") as handle:
        for row in rows:
            handle.write(json.dumps({"case_id": row["case_id"], "probabilities": output[row["case_id"]]}, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
