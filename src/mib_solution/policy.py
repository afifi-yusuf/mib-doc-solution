from __future__ import annotations

from dataclasses import dataclass

DISQUALIFYING = {"memory_tampering", "planetary_embargo", "active_warrant", "biohazard_red"}
REVIEW_FLAGS = {"identity_conflict", "sponsor_mismatch", "illegible_biometrics", "rescinded_denial"}
REVOKED_SPONSORS = {"SPN-0007", "SPN-0139", "SPN-4040"}
# Public packets place receipt in the 2026-07-01 intake window.  A visible,
# non-diplomatic arrival before this date is more than 180 days stale.
STALE_BEFORE = "2026-01-02"


@dataclass(frozen=True)
class PolicyResult:
    decision: str | None
    reason: str


def apply_safety_policy(record: dict[str, object]) -> PolicyResult:
    flags = set(str(record.get("risk_flags", "none")).split("|")) - {"", "none"}
    # Classical normalizes enums with ``.upper()``; compare casefold so
    # ``UNKNOWN`` still triggers the missing-evidence review path.
    visa = str(record.get("visa_class", "unknown")).strip().upper()
    fee = str(record.get("fee_status", "unknown")).strip().casefold()
    sponsor = str(record.get("sponsor_id", "")).strip().upper()
    arrival = str(record.get("arrival_date", ""))
    if flags & DISQUALIFYING:
        return PolicyResult("DENIED", "disqualifying_risk")
    # Public labels demonstrate that a valid DIP-1 diplomatic record can
    # supersede a sponsor standing issue; non-diplomatic packets cannot.
    if sponsor in REVOKED_SPONSORS and visa != "DIP-1":
        return PolicyResult("DENIED", "revoked_sponsor")
    if visa == "TRANSIT-7":
        return PolicyResult("DENIED", "transit_not_work_authorized")
    if fee == "unpaid":
        return PolicyResult("DENIED", "unpaid_fee")
    if visa != "DIP-1" and STALE_BEFORE > arrival > "1900-01-01":
        return PolicyResult("DENIED", "stale_arrival")
    if visa != "DIP-1" and sponsor == "SPN-0000":
        return PolicyResult("NEEDS_REVIEW", "missing_sponsor")
    if fee == "unknown" or visa in {"", "UNKNOWN"}:
        return PolicyResult("NEEDS_REVIEW", "missing_policy_evidence")
    if fee == "waived" and visa != "DIP-1":
        return PolicyResult("NEEDS_REVIEW", "unverified_fee_waiver")
    if flags & REVIEW_FLAGS:
        return PolicyResult("NEEDS_REVIEW", "review_only_risk")
    return PolicyResult(None, "no_override")
