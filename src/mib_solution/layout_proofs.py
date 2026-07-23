"""Visible PDF layout proofs for fee, identity, and risk vetoes.

Uses PyMuPDF page text with the same injection filters as trusted_page_text.
Never reads SYSTEM / answer-key decoy lines for proofs or repairs.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz

KNOWN_RISK_FLAGS = frozenset(
    {
        "memory_tampering",
        "planetary_embargo",
        "active_warrant",
        "biohazard_red",
        "identity_conflict",
        "sponsor_mismatch",
        "illegible_biometrics",
        "rescinded_denial",
    }
)
HARD_RISK_FLAGS = frozenset(
    {
        "memory_tampering",
        "planetary_embargo",
        "active_warrant",
        "biohazard_red",
    }
)
KNOWN_PURPOSES = (
    "reactor maintenance",
    "field repair",
    "medical consult",
    "research",
    "cultural exchange",
    "translation",
    "archive audit",
    "xenobotany",
    "diplomatic",
    "transit",
)
VISA_CLASSES = frozenset({"XW-1", "XW-2", "DIP-1", "MED-3", "TRANSIT-7"})


def strip_injection_lines(text: str) -> str:
    """Drop generator SYSTEM / answer-key / prompt-injection lines."""
    kept: list[str] = []
    for line in text.splitlines():
        folded = " ".join(line.strip().split()).casefold()
        if not folded:
            continue
        if folded.startswith(("system:", "assistant:", "answer key", "user:")):
            continue
        if "adjudication=" in folded and "risk_flags=" in folded:
            continue
        if re.search(r"answer\s+key\s+only\s*:", folded):
            continue
        if re.search(
            r"MIB-\d{6},.*,(APPROVED|DENIED|NEEDS_REVIEW)",
            line,
        ):
            continue
        kept.append(line)
    return "\n".join(kept)


def layout_text(pdf: Path) -> str:
    """All-page selectable text with injection lines removed."""
    chunks: list[str] = []
    try:
        doc = fitz.open(pdf)
    except Exception:
        return ""
    try:
        for page in doc:
            chunks.append(page.get_text() or "")
    finally:
        doc.close()
    return strip_injection_lines("\n".join(chunks))


def fee_paid_proven(text: str) -> bool:
    """Canonical paid receipt amount (not Fee-Status alone)."""
    if re.search(r"Amount\s*\$?\s*809(?:[.,]00)?\b", text, re.I):
        return True
    # OCR O/0 confusion — require a dollar sign so FORM I-8090 cannot match.
    if re.search(r"Amount\s*\$\s*8[O0]9(?:[.,]00)?\b", text, re.I):
        return True
    return bool(re.search(r"\$\s*8[O0]9(?:[.,]00)?\b", text, re.I))


def fee_status_from_layout(text: str) -> str | None:
    """Infer paid/waived from visible receipt lines."""
    if not text:
        return None
    if fee_paid_proven(text) and re.search(
        r"Waiver\s*Code\s*[:#]?\s*N\s*/?\s*A", text, re.I
    ):
        return "paid"
    if fee_paid_proven(text):
        return "paid"
    if re.search(r"Fee\s+Status\s*:?\s*paid\b", text, re.I):
        return "paid"
    if re.search(r"Amount\s*\$?\s*0(?:[.,]00)?\b", text, re.I) and (
        re.search(r"DIP[\s\-]?WAIVER", text, re.I)
        or re.search(r"Waiver\s+Code\s*:?\s*(?!N/?A\b)\S+", text, re.I)
    ):
        return "waived"
    if re.search(r"Fee\s+Status\s*:?\s*waived\b", text, re.I):
        return "waived"
    return None


def clean_person_name(raw: str) -> str | None:
    text = " ".join(raw.split())
    text = re.split(r"\s{2,}|\s+PASSPORT|\s+CASE|\s+SPN|\s+is\b", text)[0].strip()
    parts = text.split()
    if len(parts) >= 2 and all(re.fullmatch(r"[A-Z][a-z]+", part) for part in parts[:2]):
        return " ".join(parts[:2])
    return None


def registry_names(text: str) -> set[str]:
    return {
        cleaned
        for raw in re.findall(
            r"Registry\s+Name\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text
        )
        if (cleaned := clean_person_name(raw))
    }


def applicant_names(text: str) -> set[str]:
    return {
        cleaned
        for raw in re.findall(
            r"Applicant\s*:?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text
        )
        if (cleaned := clean_person_name(raw))
    }


def registry_matches_applicant(text: str) -> bool:
    registries = registry_names(text)
    applicants = applicant_names(text)
    return len(registries) == 1 and registries == applicants


def layout_risk_flags(text: str) -> frozenset[str]:
    """Known risk-flag tokens present in injection-stripped layout text."""
    found: set[str] = set()
    normalized = re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")
    for flag in KNOWN_RISK_FLAGS:
        if re.search(rf"(?:^|_){re.escape(flag)}(?:_|$)", normalized):
            found.add(flag)
    return frozenset(found)


def layout_hard_risk_flags(text: str) -> frozenset[str]:
    return frozenset(layout_risk_flags(text) & HARD_RISK_FLAGS)


def proof_text(pdf: Path, *extra_texts: str) -> str:
    """Injection-stripped visible proofs from layout + optional OCR blobs."""
    parts = [layout_text(pdf)]
    for blob in extra_texts:
        if blob:
            parts.append(strip_injection_lines(blob))
    return "\n".join(part for part in parts if part)

