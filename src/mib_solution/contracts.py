from __future__ import annotations

from datetime import date

FIELDNAMES = (
    "case_id", "applicant_name", "species_code", "home_world", "visa_class",
    "sponsor_id", "arrival_date", "declared_purpose", "risk_flags", "fee_status",
    "adjudication", "confidence",
)
FEE_VALUES = {"paid", "waived", "unpaid", "unknown"}
DECISIONS = {"APPROVED", "DENIED", "NEEDS_REVIEW"}


def blank_record(case_id: str) -> dict[str, object]:
    """Return a schema-valid conservative record when evidence is unavailable."""
    return {
        "case_id": case_id,
        "applicant_name": "unknown",
        "species_code": "unknown",
        "home_world": "unknown",
        "visa_class": "unknown",
        "sponsor_id": "SPN-0000",
        "arrival_date": "1900-01-01",
        "declared_purpose": "unknown",
        "risk_flags": "none",
        "fee_status": "unknown",
        "adjudication": "NEEDS_REVIEW",
        "confidence": 0.01,
    }


def normalize_flags(values: set[str] | list[str]) -> str:
    return "|".join(sorted(set(values))) if values else "none"


def is_iso_date(value: str) -> bool:
    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False

