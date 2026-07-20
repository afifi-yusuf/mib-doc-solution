from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def render_pdf(pdf_path: Path, output_dir: Path, dpi: int = 180) -> list[Path]:
    """Render only the visible PDF page content; do not consume its text layer."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "page"
    existing = sorted(output_dir.glob("page-*.png"))
    if existing:
        return existing
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), str(prefix)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    return sorted(output_dir.glob("page-*.png"))


def variants(image_path: Path) -> list[tuple[str, Image.Image]]:
    """Create bounded OCR retries without changing the original visual evidence."""
    original = Image.open(image_path).convert("RGB")
    gray = ImageOps.grayscale(original)
    contrast = ImageEnhance.Contrast(gray).enhance(1.8)
    clean = contrast.filter(ImageFilter.MedianFilter(size=3))
    return [("original", original), ("contrast", contrast), ("clean", clean)]

