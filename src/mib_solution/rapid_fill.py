"""Fail-soft RapidOCR fill for unknown fee/risk/visa fields only.

Never overrides trusted text-layer values. Any RapidOCR import or runtime
failure returns empty fills so the primary Tesseract pipeline stays intact.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import fitz
import numpy as np

_ENGINE = None
_ENGINE_LOCK = threading.Lock()
_ENGINE_FAILED = False


@dataclass(frozen=True)
class RapidFill:
    fee_status: str | None = None
    visa_class: str | None = None
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

            os.environ.setdefault("OMP_THREAD_LIMIT", "1")
            _ENGINE = RapidOCR()
        except Exception:
            _ENGINE_FAILED = True
            return None
    return _ENGINE


def _page_text(engine, page: fitz.Page) -> str:
    pix = page.get_pixmap(dpi=140, alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        image = image[:, :, :3]
    try:
        result = engine(image)
    except Exception:
        return ""
    texts = getattr(result, "txts", None) or ()
    return "\n".join(str(part) for part in texts if part)


def risk_cleared_from_text(text: str) -> bool:
    from .classical import risk_evidence_cleared

    return risk_evidence_cleared(text)


def parse_rapid_text(text: str) -> RapidFill:
    """Extract fee/risk/visa signals from RapidOCR page text."""
    # Lazy imports avoid a classical <-> rapid_fill cycle at module load.
    from .classical import (
        canonicalize_fee_status,
        canonicalize_visa_class,
        recover_flags_from_text,
    )

    fee = None
    visa = None
    for line in text.splitlines():
        if fee is None and (
            re.search(r"fee\s*status|^\s*status\s*:", line, re.I)
            or re.search(r"\$\s*809|\$\s*0\.00", line)
        ):
            cand = canonicalize_fee_status(line)
            if cand and cand != "unknown":
                fee = cand
        if fee is None and re.search(r"\$\s*809", line):
            fee = "paid"
        if fee is None and re.search(r"\$\s*0\.00", line):
            fee = "waived"
        if visa is None and re.search(r"visa\s*class", line, re.I):
            match = re.search(
                r"visa\s*class\s*[:=]?\s*([A-Za-z0-9-]{2,12})",
                line,
                re.I,
            )
            if match:
                visa = canonicalize_visa_class(match.group(1))
    if fee is None:
        fee = canonicalize_fee_status(text)
        if fee == "unknown":
            fee = None
    if visa is None:
        match = re.search(r"\b(XW-?[12]|MED-?3|DIP-?1|TRANSIT-?7)\b", text, re.I)
        if match:
            visa = canonicalize_visa_class(match.group(1))
    flags = recover_flags_from_text(text)
    cleared = risk_cleared_from_text(text)
    return RapidFill(
        fee_status=fee,
        visa_class=visa,
        risk_flags=flags or None,
        risk_cleared=cleared,
        text=text,
    )


def rapid_fill_unknowns(
    pdf: Path,
    *,
    page_indices: set[int],
    need_fee: bool = True,
    need_risk: bool = True,
    need_visa: bool = False,
) -> RapidFill:
    """OCR selected pages and return fills for requested unknown fields."""
    if not need_fee and not need_risk and not need_visa:
        return RapidFill()
    if not page_indices:
        return RapidFill()
    engine = _get_engine()
    if engine is None:
        return RapidFill()
    try:
        doc = fitz.open(pdf)
    except Exception:
        return RapidFill()
    chunks: list[str] = []
    try:
        for page_index in sorted(page_indices):
            if page_index < 1 or page_index > len(doc):
                continue
            chunks.append(_page_text(engine, doc[page_index - 1]))
    except Exception:
        return RapidFill()
    finally:
        doc.close()
    combined = "\n".join(chunk for chunk in chunks if chunk.strip())
    if not combined.strip():
        return RapidFill()
    parsed = parse_rapid_text(combined)
    # Always keep risk_cleared/text from the full parse; callers decide what to apply.
    return RapidFill(
        fee_status=parsed.fee_status if need_fee else None,
        visa_class=parsed.visa_class if need_visa else None,
        risk_flags=parsed.risk_flags if need_risk else None,
        risk_cleared=parsed.risk_cleared,
        text=combined,
    )
