from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from .model import PacketCNN

IMAGE_SIZE = (224, 320)  # width, height; preserves a document-like aspect ratio


def page_tensor(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    data = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(IMAGE_SIZE[1], IMAGE_SIZE[0], 3)
    return data.permute(2, 0, 1).div(255.0)


class VisionPredictor:
    def __init__(self, model_path: Path, maps_path: Path):
        self.maps = json.loads(maps_path.read_text())
        self.task_sizes = {name: len(values) for name, values in self.maps["tasks"].items()}
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model = PacketCNN(self.task_sizes, width=checkpoint.get("width", 32))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.threshold = float(self.maps.get("decision_threshold", 0.55))

    def predict(self, page_paths: list[Path]) -> tuple[dict[str, str], float]:
        pages = torch.stack([page_tensor(path) for path in page_paths[:6]])
        count = pages.shape[0]
        if count < 6:
            pages = torch.cat([pages, torch.zeros(6 - count, *pages.shape[1:])])
        mask = torch.tensor([[True] * count + [False] * (6 - count)])
        with torch.no_grad():
            output = self.model(pages.unsqueeze(0), mask)
        decoded: dict[str, str] = {}
        decision_confidence = 0.0
        for task, logits in output.items():
            probabilities = torch.softmax(logits, dim=1)[0]
            index = int(probabilities.argmax())
            decoded[task] = self.maps["tasks"][task][index]
            if task == "adjudication":
                decision_confidence = float(probabilities[index])
        return decoded, decision_confidence


def load_predictor(model_path: Path, maps_path: Path) -> VisionPredictor | None:
    if not model_path.is_file() or not maps_path.is_file():
        return None
    return VisionPredictor(model_path, maps_path)
