from __future__ import annotations

import csv
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class OCRPage:
    page: int
    variant: str
    text: str
    confidence: float


def read_tsv(image: Image.Image) -> tuple[str, float]:
    with tempfile.NamedTemporaryFile(suffix=".png") as handle:
        image.save(handle.name)
        result = subprocess.run(
            ["tesseract", handle.name, "stdout", "--psm", "6", "tsv"],
            check=False, capture_output=True, text=True,
        )
    if result.returncode != 0:
        return "", 0.0
    rows = list(csv.DictReader(result.stdout.splitlines(), delimiter="\t"))
    words = [r.get("text", "").strip() for r in rows if r.get("text", "").strip()]
    scores = []
    for row in rows:
        try:
            score = float(row.get("conf", "-1"))
        except ValueError:
            continue
        if score >= 0:
            scores.append(score)
    return " ".join(words), (sum(scores) / len(scores) / 100.0 if scores else 0.0)

