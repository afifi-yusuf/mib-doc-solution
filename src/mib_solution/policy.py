from __future__ import annotations

from dataclasses import dataclass

DISQUALIFYING = {"memory_tampering", "planetary_embargo", "active_warrant", "biohazard_red"}
REVIEW_FLAGS = {"identity_conflict", "sponsor_mismatch", "illegible_biometrics", "rescinded_denial"}
REVOKED_SPONSORS = {"SPN-0007", "SPN-0139", "SPN-4040"}


@dataclass(frozen=True)
class PolicyResult:
    decision: str | None
    reason: str


def apply_safety_policy(record: dict[str, object]) -> PolicyResult:
    flags = set(str(record.get("risk_flags", "none")).split("|")) - {"", "none"}
    visa = str(record.get("visa_class", "unknown"))
    fee = str(record.get("fee_status", "unknown"))
    sponsor = str(record.get("sponsor_id", ""))
    if flags & DISQUALIFYING:
        return PolicyResult("DENIED", "disqualifying_risk")
    if sponsor in REVOKED_SPONSORS:
        return PolicyResult("DENIED", "revoked_sponsor")
    if visa == "TRANSIT-7":
        return PolicyResult("DENIED", "transit_not_work_authorized")
    if fee == "unpaid":
        return PolicyResult("DENIED", "unpaid_fee")
    if visa != "DIP-1" and sponsor == "SPN-0000":
        return PolicyResult("NEEDS_REVIEW", "missing_sponsor")
    if fee == "unknown" or visa == "unknown":
        return PolicyResult("NEEDS_REVIEW", "missing_policy_evidence")
    if fee == "waived" and visa != "DIP-1":
        return PolicyResult("NEEDS_REVIEW", "unverified_fee_waiver")
    if flags & REVIEW_FLAGS:
        return PolicyResult("NEEDS_REVIEW", "review_only_risk")
    return PolicyResult(None, "no_override")
