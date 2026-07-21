"""Weakly supervised, vision-only detector for visible risk/stamp evidence.

Packet labels supervise a max-pooling page model: a positive packet must have
at least one positive page, but no page regions are manually annotated.
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

FLAGS = ("memory_tampering", "active_warrant", "biohazard_red")
SIZE = (128, 160)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tensor(path: Path, augment: bool = False) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize(SIZE)
    pixels = torch.from_numpy(np.asarray(image).copy()).to(torch.float32)
    value = pixels.permute(2, 0, 1).div(255.0)
    if augment:
        value = (value * float(torch.empty(1).uniform_(0.85, 1.2))).clamp(0, 1)
    return value


class PacketPages(Dataset):
    def __init__(self, rows: list[dict[str, str]], cache: Path, augment: bool = False):
        self.rows, self.cache, self.augment = rows, cache, augment
        self.tensor_cache: dict[Path, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        pages = sorted((self.cache / row["case_id"]).glob("page-*.png"))[:6]
        values = torch.stack([self._page_tensor(page) for page in pages])
        if len(pages) < 6:
            values = torch.cat([values, torch.zeros(6 - len(pages), *values.shape[1:])])
        mask = torch.tensor([True] * len(pages) + [False] * (6 - len(pages)))
        flags = set(row["risk_flags"].split("|"))
        target = torch.tensor([flag in flags for flag in FLAGS], dtype=torch.float32)
        return values, mask, target, row["case_id"]

    def _page_tensor(self, page: Path) -> torch.Tensor:
        value = self.tensor_cache.get(page)
        if value is None:
            value = tensor(page)
            self.tensor_cache[page] = value
        if self.augment:
            return (value * float(torch.empty(1).uniform_(0.85, 1.2))).clamp(0, 1)
        return value


class StampMIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2, 2), nn.BatchNorm2d(24), nn.SiLU(),
            nn.Conv2d(24, 48, 3, 2, 1), nn.BatchNorm2d(48), nn.SiLU(),
            nn.Conv2d(48, 72, 3, 2, 1), nn.BatchNorm2d(72), nn.SiLU(),
            nn.Conv2d(72, 96, 3, 2, 1), nn.BatchNorm2d(96), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(96, len(FLAGS))

    def forward(self, pages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, count, channels, height, width = pages.shape
        features = self.encoder(pages.reshape(batch * count, channels, height, width)).flatten(1)
        logits = self.head(features).reshape(batch, count, len(FLAGS))
        return logits.masked_fill(~mask.unsqueeze(-1), -30.0).amax(dim=1)


def train(model: StampMIL, loader: DataLoader, target_device: torch.device, epochs: int, weights: torch.Tensor) -> None:
    model.to(target_device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights.to(target_device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for epoch in range(epochs):
        model.train()
        losses = []
        for pages, mask, target, _ in loader:
            logits = model(pages.to(target_device), mask.to(target_device))
            loss = loss_fn(logits, target.to(target_device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"stamp epoch {epoch + 1}/{epochs}: loss={sum(losses)/max(1,len(losses)):.4f}", flush=True)


def probabilities(model: StampMIL, loader: DataLoader, target_device: torch.device) -> dict[str, dict[str, float]]:
    model.eval()
    output: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for pages, mask, _, case_ids in loader:
            values = torch.sigmoid(model(pages.to(target_device), mask.to(target_device))).cpu()
            for case_id, row in zip(case_ids, values):
                output[case_id] = {flag: float(value) for flag, value in zip(FLAGS, row)}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--oof", type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    with args.labels.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    target = device()
    print(f"Stamp CNN device: {target}", flush=True)
    labels = ["1" if any(flag in row["risk_flags"].split("|") for flag in FLAGS) else "0" for row in rows]
    positive = torch.tensor([sum(flag in row["risk_flags"].split("|") for row in rows) for flag in FLAGS], dtype=torch.float32)
    weights = (len(rows) - positive).clamp_min(1) / positive.clamp_min(1)
    if args.oof:
        output: dict[str, dict[str, float]] = {}
        folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=8090)
        for fold, (train_idx, valid_idx) in enumerate(folds.split(rows, labels), start=1):
            print(f"Stamp OOF fold {fold}/{args.folds}", flush=True)
            model = StampMIL()
            train_rows = [rows[index] for index in train_idx]
            valid_rows = [rows[index] for index in valid_idx]
            train(model, DataLoader(PacketPages(train_rows, args.cache, True), batch_size=8, shuffle=True), target, args.epochs, weights)
            output.update(probabilities(model, DataLoader(PacketPages(valid_rows, args.cache), batch_size=8), target))
        with args.oof.open("w") as handle:
            for row in rows:
                handle.write(json.dumps({"case_id": row["case_id"], "probabilities": output[row["case_id"]]}, sort_keys=True) + "\n")
    model = StampMIL()
    train(model, DataLoader(PacketPages(rows, args.cache, True), batch_size=8, shuffle=True), target, args.epochs, weights)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "flags": FLAGS, "size": SIZE}, args.artifact)
    print(f"Saved stamp CNN to {args.artifact}", flush=True)


if __name__ == "__main__":
    main()
