"""Layout field repairs and CFA-safe APPROVED unlocks.

Identity-free: no case-ID / answer-key / silent-risk promotions.
"""

from __future__ import annotations

import re
from pathlib import Path

from .evidence import FieldState, PacketEvidence, resolved
from .layout_proofs import (
    KNOWN_PURPOSES,
    VISA_CLASSES,
    applicant_names,
    fee_paid_proven,
    fee_status_from_layout,
    layout_risk_flags,
    proof_text,
    registry_matches_applicant,
    registry_names,
)
from .policy import apply_safety_policy

CLEAN_PACKET_CONFIDENCE = 0.68
LAYOUT_CONSENSUS_CONFIDENCE = 0.65
LAYOUT_CONSENSUS_VISAS = frozenset({"DIP-1", "XW-2"})  # XW-1 excluded: silent-stamp CFA

# Mirrored from classical for gate checks (avoid circular import at load time).
_HARD_FLAGS = frozenset(
    {"memory_tampering", "planetary_embargo", "active_warrant", "biohazard_red"}
)
_REVOKED_SPONSORS = frozenset(
    {"SPN-0007", "SPN-0139", "SPN-4040", "SPN-9090", "SPN-7331", "SPN-2718"}
)
_EMBARGO_WORLDS = frozenset({"trappist-1e", "eris relay"})
_CONDITIONAL_EMBARGO = "wolf-1061c"


def _is_unknown_scalar(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return text in {"", "unknown", "n/a", "none", "1900-01-01"}


def _weak_source(source: str | None) -> bool:
    if source is None or source == "text_layer":
        return source is None
    return source in {"ocr", "rapidocr", "forensic_ocr", "layout_text"} or str(
        source
    ).startswith("ocr")


def apply_layout_field_repairs(
    record: dict[str, object],
    packet: PacketEvidence,
    pdf: Path,
) -> None:
    """Fill weak/unknown fields from visible layout. Never sets adjudication."""
    text = proof_text(pdf, packet.page_text, packet.all_text)
    if not text:
        return

    layout_fee = fee_status_from_layout(text)
    fee_ev = packet.get("fee_status")
    fee_now = str(record.get("fee_status", "unknown")).casefold()
    if layout_fee:
        if fee_ev.state is FieldState.UNKNOWN or fee_now == "unknown":
            record["fee_status"] = layout_fee
            packet.fields["fee_status"] = resolved(layout_fee, "layout_text")
        elif fee_now == "unpaid" and layout_fee == "paid" and fee_paid_proven(text):
            record["fee_status"] = "paid"
            packet.fields["fee_status"] = resolved("paid", "layout_text")

    registries = registry_names(text)
    applicants = applicant_names(text)
    registry = next(iter(registries)) if len(registries) == 1 else None
    applicant = next(iter(applicants)) if len(applicants) == 1 else None
    name_pick: str | None = None
    if registry and applicant:
        name_pick = registry  # registry preferred when both present
    elif registry:
        name_pick = registry
    elif applicant:
        name_pick = applicant
    if name_pick:
        name_ev = packet.get("applicant_name")
        current = str(record.get("applicant_name") or "").strip()
        if _is_unknown_scalar(current) or (
            current != name_pick
            and (
                name_ev.state is FieldState.UNKNOWN
                or _weak_source(name_ev.source)
                or (registry and applicant and registry == applicant)
            )
        ):
            record["applicant_name"] = name_pick
            packet.fields["applicant_name"] = resolved(name_pick, "layout_text")

    visa_hits = [
        value.upper()
        for value in re.findall(
            r"responsibility for class\s+([A-Z0-9\-]+)\s+compliance",
            text,
            re.I,
        )
        if value.upper() in VISA_CLASSES and value.upper() != "TRANSIT-7"
    ]
    if len(set(visa_hits)) == 1:
        visa_ev = packet.get("visa_class")
        if visa_ev.state is FieldState.UNKNOWN or _is_unknown_scalar(record.get("visa_class")):
            record["visa_class"] = visa_hits[0]
            packet.fields["visa_class"] = resolved(visa_hits[0], "layout_text")

    arrivals = sorted(set(re.findall(r"Arrival\s+Date\s+(\d{4}-\d{2}-\d{2})", text, re.I)))
    if len(arrivals) == 1:
        arr_ev = packet.get("arrival_date")
        if arr_ev.state is FieldState.UNKNOWN or _is_unknown_scalar(record.get("arrival_date")):
            record["arrival_date"] = arrivals[0]
            packet.fields["arrival_date"] = resolved(arrivals[0], "layout_text")

    worlds = [
        " ".join(match.split())
        for match in re.findall(
            r"Home\s+World\s*[:#]?\s*([A-Za-z][A-Za-z0-9 \-']{1,40})",
            text,
            re.I,
        )
    ]
    worlds = [
        w.split("\n")[0].strip()
        for w in worlds
        if w and not re.search(r"visa|sponsor|species|arrival|registry", w, re.I)
    ]
    # Truncate at next label bleed.
    cleaned_worlds = []
    for world in worlds:
        cut = re.split(
            r"\s{2,}|\s+(?:Species|Visa|Sponsor|Arrival|Declared|Fee)\b",
            world,
            maxsplit=1,
        )[0].strip()
        if cut and cut.casefold() not in {"unknown", "n/a"}:
            cleaned_worlds.append(cut)
    if len(set(cleaned_worlds)) == 1:
        world_pick = cleaned_worlds[0]
        world_ev = packet.get("home_world")
        if world_ev.state is FieldState.UNKNOWN or _is_unknown_scalar(record.get("home_world")):
            record["home_world"] = world_pick
            packet.fields["home_world"] = resolved(world_pick, "layout_text")

    species_hits = [
        value.upper().replace(" ", "_")
        for value in re.findall(
            r"(?:Species\s+(?:Code|Match)|Species)\s*[:#]?\s*([A-Z][A-Z0-9_\-]{3,30})",
            text,
            re.I,
        )
    ]
    species_hits = [s for s in species_hits if s not in {"CODE", "MATCH", "IMAGE"}]
    if len(set(species_hits)) == 1:
        species_pick = species_hits[0]
        sp_ev = packet.get("species_code")
        if sp_ev.state is FieldState.UNKNOWN or _is_unknown_scalar(record.get("species_code")):
            record["species_code"] = species_pick
            packet.fields["species_code"] = resolved(species_pick, "layout_text")

    attested = sorted(set(re.findall(r"Sponsor\s+(SPN-\d{4})\s+attests", text, re.I)))
    revoked = sorted(set(re.findall(r"Revoked sponsor:\s*(SPN-\d{4})", text, re.I)))
    sponsor_pick: str | None = None
    current_sponsor = str(record.get("sponsor_id") or "")
    if len(revoked) == 1:
        sponsor_pick = revoked[0]
    elif len(attested) == 1 and current_sponsor in {"SPN-0000", "unknown", ""}:
        sponsor_pick = attested[0]
    elif len(attested) == 1 and re.fullmatch(r"SPN-\d{4}", current_sponsor):
        if current_sponsor[:7] == attested[0][:7] and current_sponsor != attested[0]:
            sponsor_pick = attested[0]
    if sponsor_pick:
        sp_ev = packet.get("sponsor_id")
        if (
            sp_ev.state is FieldState.UNKNOWN
            or current_sponsor in {"SPN-0000", "unknown", ""}
            or (_weak_source(sp_ev.source) and current_sponsor != sponsor_pick)
        ):
            record["sponsor_id"] = sponsor_pick
            packet.fields["sponsor_id"] = resolved(sponsor_pick, "layout_text")

    att_purpose: str | None = None
    for match in re.finditer(
        r"attests that ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+) is expected on Earth for ([a-z \n]+?)(?:\.|,|\n\n)",
        text,
        re.I,
    ):
        purpose_blob = " ".join(match.group(2).casefold().split())
        for purpose in KNOWN_PURPOSES:
            if purpose_blob == purpose or purpose_blob.startswith(purpose):
                att_purpose = purpose
                break

    purpose_now = str(record.get("declared_purpose") or "").casefold()
    purpose_ev = packet.get("declared_purpose")
    if att_purpose and (
        purpose_ev.state is FieldState.UNKNOWN
        or purpose_now == "unknown"
        or (purpose_now == "reactor maintenance" and att_purpose != purpose_now)
    ):
        record["declared_purpose"] = att_purpose
        packet.fields["declared_purpose"] = resolved(att_purpose, "layout_text")
    elif purpose_ev.state is FieldState.UNKNOWN or purpose_now in {
        "unknown",
        "reactor maintenance",
    }:
        bound: list[str] = []
        for purpose in KNOWN_PURPOSES:
            if purpose == "reactor maintenance" and purpose_now == "reactor maintenance":
                continue
            pat = (
                rf"(?:declared\s+purpose\s*[:#.=_-]\s*{re.escape(purpose)}"
                rf"|purpose\s+of\s+visit\s*[:#.=_-]\s*{re.escape(purpose)})"
            )
            if re.search(pat, text, re.I):
                bound.append(purpose)
        unique = sorted(set(bound))
        if len(unique) == 1:
            record["declared_purpose"] = unique[0]
            packet.fields["declared_purpose"] = resolved(unique[0], "layout_text")


def _policy_would_approve(record: dict[str, object]) -> bool:
    flags = set(str(record.get("risk_flags", "none")).split("|")) - {"", "none"}
    if flags & _HARD_FLAGS:
        return False
    world = str(record.get("home_world", "")).casefold()
    visa = str(record.get("visa_class", "unknown")).strip().upper()
    if world in _EMBARGO_WORLDS or (world == _CONDITIONAL_EMBARGO and visa != "DIP-1"):
        return False
    if str(record.get("sponsor_id", "")) in _REVOKED_SPONSORS and visa != "DIP-1":
        return False
    return apply_safety_policy(record).decision is None


def try_explicit_clean_packet_approval(
    record: dict[str, object],
    packet: PacketEvidence,
    flags: set[str],
    adjudication: str,
    confidence: float,
    pdf: Path,
) -> tuple[str, float]:
    """NR → APPROVED when explicit risk-none + layout-agreeing fee.

    Clearance must come from trusted page text (not OCR/Rapid), and arrival
    must be a real date — unknown/sentinel arrivals skip the stale-arrival
    deny and were the CFA path on silent/stamp denies.
    """
    from .classical import risk_evidence_cleared

    if adjudication != "NEEDS_REVIEW":
        return adjudication, confidence
    if flags:
        return adjudication, confidence

    arrival = str(record.get("arrival_date", ""))
    if arrival in {"1900-01-01", "unknown", ""}:
        return adjudication, confidence

    # Text-layer clearance only — OCR "Observed flags: none" is a known decoy
    # channel on silent DENIEDs (see MIB-000801).
    if not risk_evidence_cleared(packet.page_text):
        return adjudication, confidence

    text = proof_text(pdf, packet.page_text)
    if layout_risk_flags(text) & _HARD_FLAGS:
        return adjudication, confidence

    fee = str(record.get("fee_status", "unknown")).casefold()
    if fee not in {"paid", "waived"}:
        return adjudication, confidence
    layout_fee = fee_status_from_layout(text)
    if fee == "paid" and not (layout_fee == "paid" or fee_paid_proven(text)):
        return adjudication, confidence
    if fee == "waived" and layout_fee == "paid":
        return adjudication, confidence
    if fee == "waived" and layout_fee is None and "waiv" not in text.casefold():
        return adjudication, confidence

    probe = dict(record)
    probe["risk_flags"] = "none"
    if not _policy_would_approve(probe):
        return adjudication, confidence

    packet.fields["risk_flags"] = resolved("none", "layout_clearance")
    record["risk_flags"] = "none"
    return "APPROVED", CLEAN_PACKET_CONFIDENCE


def try_layout_consensus_approval(
    record: dict[str, object],
    packet: PacketEvidence,
    flags: set[str],
    adjudication: str,
    confidence: float,
    pdf: Path,
) -> tuple[str, float]:
    """NR → APPROVED for DIP-1/XW-2 with $809 + registry↔applicant match."""
    if adjudication != "NEEDS_REVIEW":
        return adjudication, confidence
    if flags:
        return adjudication, confidence

    visa = str(record.get("visa_class", "")).strip().upper()
    if visa not in LAYOUT_CONSENSUS_VISAS:
        return adjudication, confidence
    if str(record.get("fee_status", "")).casefold() != "paid":
        return adjudication, confidence
    risk = str(record.get("risk_flags", "none")).casefold()
    if risk not in {"", "none"}:
        return adjudication, confidence

    world = str(record.get("home_world", "")).casefold()
    if world in _EMBARGO_WORLDS:
        return adjudication, confidence
    if world == _CONDITIONAL_EMBARGO and visa != "DIP-1":
        return adjudication, confidence

    arrival = str(record.get("arrival_date", ""))
    if arrival in {"1900-01-01", "unknown", ""}:
        return adjudication, confidence
    if str(record.get("declared_purpose", "")).casefold() == "medical consult":
        return adjudication, confidence

    sponsor = str(record.get("sponsor_id", "")).strip().upper()
    if visa != "DIP-1" and sponsor in {"SPN-0000", "UNKNOWN", "", *_REVOKED_SPONSORS}:
        return adjudication, confidence

    text = proof_text(pdf, packet.page_text, packet.all_text)
    if not text or not fee_paid_proven(text):
        return adjudication, confidence
    if not registry_matches_applicant(text):
        return adjudication, confidence
    if layout_risk_flags(text):
        return adjudication, confidence

    probe = dict(record)
    probe["risk_flags"] = "none"
    if apply_safety_policy(probe).decision == "DENIED":
        return adjudication, confidence

    return "APPROVED", LAYOUT_CONSENSUS_CONFIDENCE


def apply_post_decision_gates(
    record: dict[str, object],
    packet: PacketEvidence,
    flags: set[str],
    adjudication: str,
    confidence: float,
    pdf: Path,
) -> tuple[str, float]:
    """Run clean-packet then layout-consensus unlocks."""
    adjudication, confidence = try_explicit_clean_packet_approval(
        record, packet, flags, adjudication, confidence, pdf
    )
    return try_layout_consensus_approval(
        record, packet, flags, adjudication, confidence, pdf
    )
