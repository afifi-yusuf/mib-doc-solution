from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .model import PacketCNN
from .render import render_pdf
from .vision_io import page_tensor

TASKS = ("adjudication", "visa_class", "fee_status", "risk_flags", "species_code", "home_world", "declared_purpose")
TASK_WEIGHTS = {"adjudication": 2.5, "risk_flags": 1.5, "visa_class": 1.0, "fee_status": 1.0,
                "species_code": 0.75, "home_world": 0.75, "declared_purpose": 0.5}


def training_device() -> torch.device:
    """Prefer local accelerators for training; submission inference remains CPU-only in Docker."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PacketDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], page_paths: dict[str, list[Path]], maps: dict[str, list[str]]):
        self.rows = rows
        self.page_paths = page_paths
        self.maps = maps
        self.indices = {task: {value: index for index, value in enumerate(values)} for task, values in maps.items()}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        paths = self.page_paths[row["case_id"]][:6]
        pages = torch.stack([page_tensor(path) for path in paths])
        if len(paths) < 6:
            pages = torch.cat([pages, torch.zeros(6 - len(paths), *pages.shape[1:])])
        mask = torch.tensor([True] * len(paths) + [False] * (6 - len(paths)))
        targets = {task: torch.tensor(self.indices[task][row[task]]) for task in TASKS}
        return pages, mask, targets


def render_training_pages(rows: list[dict[str, str]], pdf_dir: Path, cache: Path) -> dict[str, list[Path]]:
    page_paths: dict[str, list[Path]] = {}
    for index, row in enumerate(rows, start=1):
        case_id = row["case_id"]
        pdf_path = pdf_dir / f"{case_id}.pdf"
        if not pdf_path.is_file():
            raise FileNotFoundError(f"Missing training PDF: {pdf_path}")
        page_paths[case_id] = render_pdf(pdf_path, cache / case_id)
        if index % 100 == 0:
            print(f"Rendered {index}/{len(rows)} packets")
    return page_paths


def make_maps(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    return {task: sorted({row[task] for row in rows}) for task in TASKS}


def train_model(model: PacketCNN, loader: DataLoader, epochs: int, device: torch.device) -> None:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for pages, mask, targets in loader:
            pages, mask = pages.to(device), mask.to(device)
            targets = {task: value.to(device) for task, value in targets.items()}
            outputs = model(pages, mask)
            loss = sum(TASK_WEIGHTS[task] * criterion(outputs[task], targets[task]) for task in TASKS)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
        print(f"epoch {epoch + 1}/{epochs}: loss={total / max(1, len(loader)):.4f}")


def decision_points(truth: int, predicted: int, review_index: int) -> float:
    if truth == predicted:
        return 8.0
    if truth != review_index and predicted == review_index:
        return 2.0
    if truth == review_index:
        return 1.0
    return 0.0


def choose_threshold(probabilities: list[torch.Tensor], labels: list[int], review_index: int) -> float:
    best_threshold, best_score = 0.55, -1.0
    for threshold in [round(step / 100, 2) for step in range(35, 91, 5)]:
        score = 0.0
        for probs, truth in zip(probabilities, labels):
            predicted = int(probs.argmax())
            if float(probs[predicted]) < threshold:
                predicted = review_index
            score += decision_points(truth, predicted, review_index)
        if score > best_score:
            best_threshold, best_score = threshold, score
    print(f"OOF selected decision threshold {best_threshold:.2f}; raw classification {best_score:.1f}")
    return best_threshold


def oof_threshold(rows: list[dict[str, str]], page_paths: dict[str, list[Path]], maps: dict[str, list[str]],
                  folds: int, batch_size: int, device: torch.device) -> float:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=8090)
    labels = [row["adjudication"] for row in rows]
    probabilities: list[torch.Tensor] = []
    truth: list[int] = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(rows, labels), start=1):
        print(f"OOF fold {fold}/{folds}")
        train_rows = [rows[index] for index in train_idx]
        valid_rows = [rows[index] for index in valid_idx]
        model = PacketCNN({task: len(values) for task, values in maps.items()})
        train_model(model, DataLoader(PacketDataset(train_rows, page_paths, maps), batch_size=batch_size, shuffle=True), 3, device)
        model.eval()
        with torch.no_grad():
            for pages, mask, targets in DataLoader(PacketDataset(valid_rows, page_paths, maps), batch_size=batch_size):
                logits = model(pages.to(device), mask.to(device))["adjudication"].cpu()
                probabilities.extend(torch.softmax(logits, 1))
                truth.extend(targets["adjudication"].tolist())
    return choose_threshold(probabilities, truth, maps["adjudication"].index("NEEDS_REVIEW"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train compact custom packet CNN on public MIB training data")
    parser.add_argument("--train-pdfs", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=8090)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    with args.labels.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    maps = make_maps(rows)
    page_paths = render_training_pages(rows, args.train_pdfs, args.cache)
    device = training_device()
    print(f"Training device: {device}")
    threshold = oof_threshold(rows, page_paths, maps, args.folds, args.batch_size, device)
    model = PacketCNN({task: len(values) for task, values in maps.items()})
    train_model(model, DataLoader(PacketDataset(rows, page_paths, maps), batch_size=args.batch_size, shuffle=True), args.epochs, device)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "width": 32}, args.artifacts / "model.pt")
    (args.artifacts / "label_maps.json").write_text(json.dumps({"tasks": maps, "decision_threshold": threshold}, indent=2) + "\n")
    print(f"Saved artifacts to {args.artifacts}")


if __name__ == "__main__":
    main()
