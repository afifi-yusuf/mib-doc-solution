from __future__ import annotations

import csv
import os
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


@dataclass(frozen=True)
class OCRLine:
    """A locally-OCR'd visual line, retaining enough geometry for field pairing."""
    text: str
    confidence: float
    x0: int
    y0: int
    x1: int
    y1: int


def read_tsv(image: Image.Image) -> tuple[str, float]:
    lines = read_lines(image)
    text = " ".join(line.text for line in lines)
    confidence = sum(line.confidence for line in lines) / len(lines) if lines else 0.0
    return text, confidence


def read_lines(image: Image.Image) -> list[OCRLine]:
    """Run local Tesseract and group words into physical lines."""
    with tempfile.NamedTemporaryFile(suffix=".png") as handle:
        image.save(handle.name)
        result = subprocess.run(
            ["tesseract", handle.name, "stdout", "--psm", "6", "tsv"],
            check=False, capture_output=True, text=True,
            # A packet worker owns one CPU.  Without this cap, several
            # concurrent Tesseract calls oversubscribe the 4-vCPU container.
            env={**os.environ, "OMP_THREAD_LIMIT": "1"},
        )
    if result.returncode != 0:
        return []
    rows = list(csv.DictReader(result.stdout.splitlines(), delimiter="\t"))
    grouped: dict[tuple[str, str, str, str], list[tuple[str, float, int, int, int, int]]] = {}
    for row in rows:
        word = row.get("text", "").strip()
        if not word:
            continue
        try:
            score = float(row.get("conf", "-1"))
        except ValueError:
            continue
        if score < 0:
            continue
        try:
            x0, y0 = int(row["left"]), int(row["top"])
            x1, y1 = x0 + int(row["width"]), y0 + int(row["height"])
        except (KeyError, ValueError):
            continue
        key = tuple(row.get(name, "") for name in ("page_num", "block_num", "par_num", "line_num"))
        grouped.setdefault(key, []).append((word, score / 100.0, x0, y0, x1, y1))
    output: list[OCRLine] = []
    for words in grouped.values():
        words.sort(key=lambda word: word[2])
        output.append(OCRLine(
            text=" ".join(word[0] for word in words),
            confidence=sum(word[1] for word in words) / len(words),
            x0=min(word[2] for word in words), y0=min(word[3] for word in words),
            x1=max(word[4] for word in words), y1=max(word[5] for word in words),
        ))
    return output
