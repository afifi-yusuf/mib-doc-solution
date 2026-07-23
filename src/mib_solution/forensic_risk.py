"""Offline forensic OCR to resolve UNKNOWN risk via clearance or hard flags.

Does not implement negative-audit approvals: many gold hard flags are
stamp-only and invisible to OCR, so inventing "clean" from silence recreates CFAs.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

_ENGINE = None
_ENGINE_LOCK = threading.Lock()
_ENGINE_FAILED = False


@dataclass(frozen=True)
class ForensicRiskResult:
    risk_flags: set[str] | None = None
    risk_cleared: bool = False
    text: str = ""


def _get_engine():
    global _ENGINE, _ENGINE_FAILED
    if _ENGINE_FAILED:
        return None
    if _ENGINE is not None:
        return _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE_FAILED:
            return None
        if _ENGINE is not None:
            return _ENGINE
        try:
            from rapidocr import RapidOCR

            _ENGINE = RapidOCR()
        except Exception:
            _ENGINE_FAILED = True
            return None
    return _ENGINE


def _ocr_rgb(engine, image: Image.Image) -> str:
    arr = np.asarray(image.convert("RGB"))
    try:
        result = engine(arr)
    except Exception:
        return ""
    texts = getattr(result, "txts", None) or ()
    return "\n".join(str(part) for part in texts if part)


def _page_forensic_text(engine, page: fitz.Page, dpi: int = 180) -> str:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        arr = arr[:, :, :3]
    im = Image.fromarray(arr)
    w, h = im.size
    gray = ImageOps.grayscale(im)
    contrast = ImageEnhance.Contrast(gray).enhance(2.0).filter(ImageFilter.SHARPEN)
    lower = im.crop((0, int(h * 0.50), w, h))
    mid = im.crop((0, int(h * 0.25), w, int(h * 0.75)))
    crops = [
        im.convert("RGB"),
        contrast.convert("RGB"),
        lower.convert("RGB"),
        ImageEnhance.Contrast(ImageOps.grayscale(lower)).enhance(2.2).convert("RGB"),
        mid.convert("RGB"),
        ImageEnhance.Contrast(ImageOps.grayscale(mid)).enhance(2.0).convert("RGB"),
    ]
    chunks: list[str] = []
    seen: set[str] = set()
    for crop in crops:
        text = _ocr_rgb(engine, crop)
        key = text.strip()
        if key and key not in seen:
            seen.add(key)
            chunks.append(text)
    return "\n".join(chunks)


def forensic_resolve_risk(pdf: Path, *, page_indices: set[int]) -> ForensicRiskResult:
    """Recover Observed-flags clearance or hard flags from selected pages."""
    engine = _get_engine()
    if engine is None or not page_indices:
        return ForensicRiskResult()
    try:
        doc = fitz.open(pdf)
    except Exception:
        return ForensicRiskResult()

    chunks: list[str] = []
    try:
        for page_index in sorted(page_indices):
            if page_index < 1 or page_index > len(doc):
                continue
            page = doc[page_index - 1]
            # Prefer image / footer carriers; still allow explicit audit set.
            text = _page_forensic_text(engine, page)
            if text.strip():
                chunks.append(text)
    except Exception:
        return ForensicRiskResult()
    finally:
        doc.close()

    combined = "\n".join(chunks)
    from .classical import recover_flags_from_text, risk_evidence_cleared

    flags = recover_flags_from_text(combined)
    if "risk panel missing" in combined.casefold() and not flags:
        flags = {"illegible_biometrics"}
    cleared = risk_evidence_cleared(combined)
    return ForensicRiskResult(
        risk_flags=flags or None,
        risk_cleared=cleared,
        text=combined,
    )
