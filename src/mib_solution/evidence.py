"""Typed field evidence: schema defaults are not the same as resolved proof.

Strobl-inspired locally: ``risk_flags="none"`` in the submission row may be an
emit fallback, but APPROVED requires ``FieldState.RESOLVED`` risk evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FieldState(Enum):
    RESOLVED = "resolved"
    UNKNOWN = "unknown"
    CONTESTED = "contested"


@dataclass(frozen=True)
class FieldEvidence:
    state: FieldState
    value: str | None = None
    source: str | None = None  # text_layer | ocr:* | rapidocr | sponsor_attestation | ...


def unknown(*, value: str | None = None, source: str | None = None) -> FieldEvidence:
    return FieldEvidence(FieldState.UNKNOWN, value=value, source=source)


def resolved(value: str, source: str | None) -> FieldEvidence:
    return FieldEvidence(FieldState.RESOLVED, value=value, source=source)


def contested(value: str | None = None, source: str | None = None) -> FieldEvidence:
    return FieldEvidence(FieldState.CONTESTED, value=value, source=source)


def is_resolved(evidence: FieldEvidence | None) -> bool:
    return evidence is not None and evidence.state is FieldState.RESOLVED


def is_text_layer(evidence: FieldEvidence | None) -> bool:
    return is_resolved(evidence) and evidence is not None and evidence.source == "text_layer"


@dataclass
class PacketEvidence:
    """Resolved packet evidence used by decide(); not the submission schema."""

    fields: dict[str, FieldEvidence] = field(default_factory=dict)
    manual_finding: FieldEvidence | None = None
    page_text: str = ""
    all_text: str = ""
    missing_pages: set[int] = field(default_factory=set)
    footer_only_pages: set[int] = field(default_factory=set)
    page_count: int = 0

    def get(self, name: str) -> FieldEvidence:
        return self.fields.get(name) or unknown()

    def risk_resolved(self) -> bool:
        return is_resolved(self.get("risk_flags"))

    def trusted_finding_approved(self) -> bool:
        finding = self.manual_finding
        return bool(
            finding is not None
            and finding.state is FieldState.RESOLVED
            and finding.value == "APPROVED"
            and finding.source == "text_layer"
        )
