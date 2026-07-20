from __future__ import annotations

import re
from dataclasses import dataclass

from .contracts import normalize_flags
from .ocr import OCRPage

RISK_FLAGS = {
    "memory_tampering", "planetary_embargo", "active_warrant", "biohazard_red",
    "identity_conflict", "sponsor_mismatch", "illegible_biometrics", "rescinded_denial",
}
SIMPLE_FIELDS = {
    "visa_class": r"\b(XW-[12]|DIP-1|MED-3|TRANSIT-7)\b",
    "sponsor_id": r"\bSPN-\d{4}\b",
    "arrival_date": r"\b\d{4}-\d{2}-\d{2}\b",
}
LABELS = {
    "applicant_name": ("applicant name", "applicant", "name"),
    "species_code": ("species code", "species"),
    "home_world": ("home world", "world"),
    "declared_purpose": ("declared purpose", "purpose"),
    "fee_status": ("fee status", "fee"),
}


@dataclass(frozen=True)
class Candidate:
    value: str
    page: int
    confidence: float
    variant: str


def _label_value(text: str, labels: tuple[str, ...]) -> str | None:
    # Tesseract TSV is flattened below, so stop at the next form label rather than
    # trusting line breaks that may not survive a damaged scan.
    all_labels = r"applicant(?: name)?|name|species(?: code)?|home world|world|declared purpose|purpose|fee status|fee|visa class|sponsor(?: id)?|arrival date|risk flags?"
    for label in labels:
        match = re.search(
            rf"{re.escape(label)}\s*[:#-]\s*(.+?)(?=\s+(?:{all_labels})\s*[:#-]|$)",
            text,
            re.I,
        )
        if match:
            return " ".join(match.group(1).strip().split())
    return None


def extract(ocr_pages: list[OCRPage]) -> tuple[dict[str, str], dict[str, list[Candidate]]]:
    candidates: dict[str, list[Candidate]] = {key: [] for key in SIMPLE_FIELDS | LABELS.keys()}
    risk_values: set[str] = set()
    for page in ocr_pages:
        for field, pattern in SIMPLE_FIELDS.items():
            for match in re.finditer(pattern, page.text, re.I):
                candidates[field].append(Candidate(match.group(0).upper(), page.page, page.confidence, page.variant))
        for field, labels in LABELS.items():
            value = _label_value(page.text, labels)
            if value:
                candidates[field].append(Candidate(value, page.page, page.confidence, page.variant))
        normalized = page.text.casefold().replace("-", "_").replace(" ", "_")
        risk_values.update(flag for flag in RISK_FLAGS if flag in normalized)

    result: dict[str, str] = {}
    for field, values in candidates.items():
        if not values:
            continue
        # Prefer the first high-confidence original rendering; retries only rescue missing text.
        values.sort(key=lambda item: (item.variant != "original", -item.confidence, item.page))
        result[field] = values[0].value
    result["risk_flags"] = normalize_flags(risk_values)
    return result, candidates
