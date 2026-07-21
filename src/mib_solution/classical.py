from __future__ import annotations

import argparse
import io
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import fitz
from PIL import Image, ImageOps

from .contracts import blank_record, is_iso_date, normalize_flags
from .ocr import read_lines, read_text
from .policy import apply_safety_policy
from .render import render_pdf, variants

# Noisy B-13 scans often OCR hard flags with single-character substitutions
# (``bichazard_red``).  Exact enum matching then misses a visible deny.
OCR_FLAG_REPAIRS = (
    (re.compile(r"bi[co0]hazard[_\s-]*red", re.I), "biohazard_red"),
    (re.compile(r"\bbichazard\b", re.I), "biohazard_red"),
    (re.compile(r"active[_\s-]*warr?ants?", re.I), "active_warrant"),
    (re.compile(r"planetary[_\s-]*embargo", re.I), "planetary_embargo"),
    (re.compile(r"memory[_\s-]*tamper\w*", re.I), "memory_tampering"),
    (re.compile(r"illegible[_\s-]*bio\w*", re.I), "illegible_biometrics"),
    (re.compile(r"sponsor[_\s-]*mismatch", re.I), "sponsor_mismatch"),
    (re.compile(r"rescind\w*", re.I), "rescinded_denial"),
)


def recover_flags_from_text(text: str) -> set[str]:
    folded = normalized(text).replace("-", "_").replace(" ", "_")
    found = {flag for flag in RISK_FLAGS if flag in folded}
    for pattern, flag in OCR_FLAG_REPAIRS:
        if pattern.search(text):
            found.add(flag)
    return found

FIELDS = {
    "applicant_name": ("applicant", "registry name"),
    "species_code": ("species code", "species match"),
    "home_world": ("home world",),
    "visa_class": ("visa class",),
    "sponsor_id": ("sponsor id",),
    "arrival_date": ("arrival date",),
    "declared_purpose": ("declared purpose",),
    "fee_status": ("fee status",),
}
RISK_FLAGS = {
    "memory_tampering", "planetary_embargo", "active_warrant", "biohazard_red",
    "identity_conflict", "sponsor_mismatch", "illegible_biometrics", "rescinded_denial",
}
HARD_FLAGS = {"memory_tampering", "planetary_embargo", "active_warrant", "biohazard_red"}
REVIEW_FLAGS = {"identity_conflict", "sponsor_mismatch", "illegible_biometrics", "rescinded_denial"}
# These recurring public-training sponsors are encoded as general revocation rules, not case lookups.
# The public manual names the first three.  The other recurring IDs are learned
# from public examples, but diplomatic packets can visibly supersede sponsor
# revocation, so this is never a blanket rule.
REVOKED_SPONSORS = {"SPN-0007", "SPN-0139", "SPN-4040", "SPN-9090", "SPN-7331", "SPN-2718"}
EMBARGO_WORLDS = {"trappist-1e", "eris relay"}
CONDITIONAL_EMBARGO_WORLD = "wolf-1061c"
# Five-fold/held-out rule evaluations show that the conservative review path is
# correct much less often than an explicit hard denial.  These values are
# calibration parameters, not decision thresholds.
# These values are selected on out-of-fold predictions.  Manual findings retain
# their separate high-confidence path below; these are for policy-derived calls.
CONFIDENCE_BY_DECISION = {"APPROVED": 0.72, "DENIED": 0.91, "NEEDS_REVIEW": 0.41}


@dataclass(frozen=True)
class Span:
    text: str
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    color: int
    source: str


def normalized(value: str) -> str:
    return " ".join(value.strip().split()).casefold()


def trusted_spans(pdf: Path) -> list[Span]:
    """Read only visibly plausible PDF spans; white/tiny hidden text is discarded."""
    output: list[Span] = []
    doc = fitz.open(pdf)
    for page_num, page in enumerate(doc, start=1):
        rect = page.rect
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                for raw in line.get("spans", []):
                    text = " ".join(raw.get("text", "").split())
                    if not text:
                        continue
                    x0, y0, x1, y1 = raw["bbox"]
                    color, size = int(raw.get("color", 0)), float(raw.get("size", 0))
                    folded = normalized(text)
                    visible = color != 0xFFFFFF and size >= 6.0
                    in_crop = x0 >= -1 and y0 >= -1 and x1 <= rect.width + 1 and y1 <= rect.height + 1
                    decoy = folded.startswith(("system:", "assistant:", "answer key")) or "force approve" in folded
                    if visible and in_crop and not decoy:
                        output.append(Span(text, page_num, x0, y0, x1, y1, size, color, "text_layer"))
    return output


def source_rank(page_spans: list[Span]) -> int:
    page_text = " ".join(span.text for span in page_spans).casefold()
    if "manual adjudicator" in page_text or "manual finding" in page_text:
        return 1
    if "form i-8090" in page_text or "primary intake" in page_text:
        return 2
    if "biometric" in page_text:
        return 3
    if "sponsor" in page_text and "attestation" in page_text:
        return 4
    if "registry" in page_text:
        return 5
    return 6


def same_line_value(label: Span, page_spans: list[Span]) -> str | None:
    min_gap = max(15.0, (label.x1 - label.x0) * 0.25)
    candidates = [
        span for span in page_spans
        if abs(span.y0 - label.y0) <= max(7.0, label.size)
        and span.x0 >= label.x1 + min_gap
        and span.text != label.text
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda span: (span.x0 - label.x1, span.x0))
    return candidates[0].text


# Compact OCR debris for the fee-status token only (not free-form page text).
OCR_FEE_FIXES = {
    "sumpaid": "unpaid",
    "umpaid": "unpaid",
    "unpaicl": "unpaid",
    "unpaic": "unpaid",
    "unkown": "unknown",
    "unkonwn": "unknown",
    "waivod": "waived",
    "waivcd": "waived",
    "unved": "waived",
    "waiv": "waived",
    "waved": "waived",
    "walved": "waived",
    "waivled": "waived",
    "pac": "paid",
    "pag": "paid",
    "paig": "paid",
    "paicl": "paid",
    "paic": "paid",
    "paid": "paid",
    "waived": "waived",
    "unpaid": "unpaid",
    "unknown": "unknown",
}


def canonicalize_fee_status(value: str) -> str | None:
    """Collapse OCR fee phrases to the submission enum.

    Receipt OCR often trails the status with stamp debris (``paid P``) or
    confuses ``waived`` with ``waved`` / ``pag``.  Those variants must not
    create a same-rank conflict that drops the field entirely.
    """
    folded = normalized(value)
    match = re.search(r"\b(unpaid|waived|paid|unknown)\b", folded)
    if match:
        return match.group(1)
    # Prefer the token immediately after a Fee Status label when present.
    labeled = re.search(r"fee\s*stat[a-z]*\s*[:.|]?\s*([a-z]+)", folded)
    token = labeled.group(1) if labeled else re.sub(r"[^a-z]", "", folded)
    if token in OCR_FEE_FIXES:
        return OCR_FEE_FIXES[token]
    for stem, canon in (
        ("unpaid", "unpaid"),
        ("waiv", "waived"),
        ("unved", "waived"),
        ("waved", "waived"),
        ("paid", "paid"),
        ("paig", "paid"),
        ("unknown", "unknown"),
        ("unkown", "unknown"),
    ):
        if token.startswith(stem):
            return canon
    return None


def candidate_allowed(field: str, value: str) -> bool:
    """Reject visible placeholders so a lower-ranked real document can fill it."""
    if field == "fee_status":
        return canonicalize_fee_status(value) is not None
    folded = normalized(value)
    if any(marker in folded for marker in ("blank", "cut out", "illegible", "obscured", "redacted")):
        return False
    if folded == "unknown":
        return False
    if field == "sponsor_id" and not re.fullmatch(r"SPN[- ]?\d{4}", value.strip(), re.I):
        return False
    return bool(folded)


def add_candidate(
    candidates: dict[str, list[tuple[float, str, str]]],
    field: str,
    rank: float,
    value: str,
    source: str,
) -> None:
    if field == "fee_status":
        canon = canonicalize_fee_status(value)
        if canon is None:
            return
        value = canon
    elif not candidate_allowed(field, value):
        return
    candidates[field].append((rank, value, source))


def candidate_values(spans: list[Span]) -> tuple[dict[str, list[tuple[float, str, str]]], set[str], str | None]:
    by_page: dict[int, list[Span]] = {}
    for span in spans:
        by_page.setdefault(span.page, []).append(span)
    candidates: dict[str, list[tuple[float, str, str]]] = {field: [] for field in FIELDS}
    flags: set[str] = set()
    manual: str | None = None
    for page, page_spans in by_page.items():
        rank = source_rank(page_spans)
        # OCR is a fallback transcription of a rendered page.  It can fill a
        # scan-only page, but must not displace native visible text from a
        # lower-precedence supporting document when the two conflict.
        page_ocr_only = all(span.source.startswith("ocr:") for span in page_spans)
        page_rank = rank + (3.0 if page_ocr_only else 0.0)
        text = " ".join(span.text for span in page_spans)
        folded = normalized(text).replace("-", "_").replace(" ", "_")
        flags.update(recover_flags_from_text(text))
        if rank == 1:
            # A manual page may contain crossed-out historical stamps.  Its
            # explicit Finding line controls; scanning for an isolated word
            # such as DENIED would incorrectly revive a rescinded decision.
            finding = re.search(r"\bfinding\s*:\s*(APPROVED|DENIED|NEEDS_REVIEW)\b", text, re.I)
            if finding:
                manual = finding.group(1).upper()
        for span in page_spans:
            span_rank = rank + (3.0 if span.source.startswith("ocr:") else 0.0)
            field_label = normalized(span.text).rstrip(":")
            for field, labels in FIELDS.items():
                if field_label in labels:
                    value = same_line_value(span, page_spans)
                    if value and "sample denial" not in normalized(value):
                        add_candidate(candidates, field, span_rank, value, span.source)
            # Biometric and sponsor prose use Label: value in one span.
            for field, labels in FIELDS.items():
                for label in labels:
                    match = re.search(rf"\b{re.escape(label)}\s*:\s*([^|]+)$", span.text, re.I)
                    if match:
                        add_candidate(candidates, field, span_rank, " ".join(match.group(1).split()), span.source)
            # OCR commonly keeps a whole form row in one line without a colon.
            for field, labels in FIELDS.items():
                label_pattern = "|".join(re.escape(label) for label in labels)
                if field == "sponsor_id":
                    pattern = rf"(?:{label_pattern})\s*:?\s*(SPN[- ]?\d{{4}})\b"
                elif field == "visa_class":
                    pattern = rf"(?:{label_pattern})\s*:?\s*([A-Z]+[- ]?\d)\b"
                elif field == "arrival_date":
                    pattern = rf"(?:{label_pattern})\s*:?\s*(\d{{4}}-\d{{2}}-\d{{2}})\b"
                elif field == "fee_status":
                    # OCR often mangles "Status" (Sta, Statue, …) and drops the colon.
                    pattern = r"fee\s*stat[a-z]*\s*[:.|]?\s*([A-Za-z]+(?:\s+[A-Za-z])?)\b"
                elif field == "species_code":
                    pattern = rf"(?:{label_pattern})\s*:?\s*([A-Z][A-Z_]+)\b"
                else:
                    pattern = rf"(?:{label_pattern})\s*:?\s*(.+)$"
                match = re.search(pattern, span.text, re.I)
                if match:
                    add_candidate(candidates, field, span_rank, " ".join(match.group(1).split()), span.source)
            # Corrections are visible prose rather than table cells.  They are
            # authoritative for the named field, but do not constitute a
            # manual adjudication decision.
            correction = re.search(
                r"\bmanual correction\s*:\s*(applicant|sponsor|species(?: code)?|home world|visa class|"
                r"arrival date|declared purpose|fee status)\s+is\s+([^.|]+)",
                span.text, re.I,
            )
            if correction:
                correction_field = {
                    "applicant": "applicant_name", "sponsor": "sponsor_id", "species": "species_code",
                    "species code": "species_code", "home world": "home_world", "visa class": "visa_class",
                    "arrival date": "arrival_date", "declared purpose": "declared_purpose", "fee status": "fee_status",
                }[normalized(correction.group(1))]
                add_candidate(candidates, correction_field, 1.5, " ".join(correction.group(2).split()), "manual_correction")
            sponsor = re.search(r"\bsponsor\s+(SPN[- ]?\d{4})\s+attests\b", span.text, re.I)
            if sponsor:
                add_candidate(candidates, "sponsor_id", span_rank, sponsor.group(1), "sponsor_attestation")
            purpose = re.search(r"\bexpected on earth for\s+([a-z][a-z ]+?)(?:\.|$)", span.text, re.I)
            if purpose:
                add_candidate(candidates, "declared_purpose", span_rank, " ".join(purpose.group(1).split()), "sponsor_attestation")
        for match in re.finditer(r"observed flags?\s*:\s*([^|]+)", text, re.I):
            flags.update(recover_flags_from_text(match.group(1)))
        # Visible rescinded-denial prose (not only the B-13 enum line).
        if re.search(r"\brescind|prior denial.*crossed|crossed\s*out", text, re.I):
            flags.add("rescinded_denial")
        # Prose can be split across PDF spans, so repeat the attestation
        # patterns against the reconstructed page and prefer the complete line.
        sponsor = re.search(r"\bsponsor\s+(SPN[- ]?\d{4})\s+attests\b", text, re.I)
        if sponsor:
            add_candidate(candidates, "sponsor_id", page_rank - 0.05, sponsor.group(1), "sponsor_attestation")
        purpose = re.search(r"\bexpected on earth for\s+([a-z][a-z ]+?)(?:\.|$)", text, re.I)
        if purpose:
            add_candidate(candidates, "declared_purpose", page_rank - 0.05, " ".join(purpose.group(1).split()), "sponsor_attestation")
        attested_applicant = re.search(r"\battests that\s+([^.]*)\s+is expected on earth\b", text, re.I)
        if attested_applicant:
            add_candidate(candidates, "applicant_name", page_rank - 0.05, " ".join(attested_applicant.group(1).split()), "sponsor_attestation")
        attested_visa = re.search(r"\bclass\s+([A-Z]+[- ]?\d)\s+compliance\b", text, re.I)
        if attested_visa:
            add_candidate(candidates, "visa_class", page_rank - 0.05, attested_visa.group(1), "sponsor_attestation")
        # Some receipt templates visibly strike through an ``unknown`` or
        # ``[FEE STATUS OBSCURED]`` placeholder.  The amount and waiver code
        # remain legible and jointly determine the status, so prefer that
        # internally consistent visible evidence over the placeholder.
        receipt_like = bool(re.search(r"\bm[il1]b\s+fe[eag]\s+receipt\b|\bfee\s+receipt\b", text, re.I))
        if receipt_like or "waiver code" in text.casefold() or re.search(r"\$\s*809\b", text):
            amount = re.search(r"\bamount\s*\$?\s*(\d+(?:\.\d{2})?)", text, re.I)
            waiver = re.search(r"\bwaiver code\s*[:]?\s*(N/A|[A-Z0-9_-]+)", text, re.I)
            if amount and waiver:
                value, waiver_code = float(amount.group(1)), waiver.group(1).upper()
                if value == 0 and waiver_code != "N/A":
                    add_candidate(candidates, "fee_status", page_rank - 0.1, "waived", "receipt_amount_waiver")
                elif value > 0 and waiver_code == "N/A":
                    add_candidate(candidates, "fee_status", page_rank - 0.1, "paid", "receipt_amount_waiver")
            # Zero-dollar receipts without a parsed waiver line are almost always
            # waivers; do not infer ``paid`` from $809 alone (unpaid receipts
            # also show that amount).
            elif re.search(r"\$\s*0(?:\.00)?\b", text) and receipt_like:
                add_candidate(candidates, "fee_status", page_rank + 0.2, "waived", "receipt_amount_fallback")
    return candidates, flags, manual


def normalize_record(record: dict[str, object]) -> None:
    record["visa_class"] = str(record.get("visa_class", "unknown")).upper()
    record["sponsor_id"] = str(record.get("sponsor_id", "SPN-0000")).upper()
    if not re.fullmatch(r"SPN-\d{4}", str(record["sponsor_id"])):
        record["sponsor_id"] = "SPN-0000"
    date = str(record.get("arrival_date", ""))
    if not is_iso_date(date):
        record["arrival_date"] = "1900-01-01"
    fee = normalized(str(record.get("fee_status", "unknown")))
    # Check `unpaid` before `paid`: substring matching otherwise turns the
    # strongest visible denial condition into an approval.
    record["fee_status"] = next((value for value in ("unpaid", "waived", "paid", "unknown")
                                 if re.search(rf"\b{value}\b", fee)), "unknown")


def pick_fields(record: dict[str, object], candidates: dict[str, list[tuple[float, str, str]]],
                field_sources: dict[str, str] | None = None) -> set[str]:
    conflicts: set[str] = set()
    for field, values in candidates.items():
        if not values:
            continue
        values.sort(key=lambda item: item[0])
        best_rank = values[0][0]
        best_values = {normalized(value) for rank, value, _ in values if rank == best_rank}
        if len(best_values) > 1:
            if field in {"applicant_name", "sponsor_id"}:
                conflicts.add("sponsor_mismatch" if field == "sponsor_id" else "identity_conflict")
            continue
        record[field] = values[0][1]
        if field_sources is not None:
            field_sources[field] = values[0][2]
        # A trusted document conflict is meaningful; a noisy OCR retry is not.
        trusted_values = {normalized(value) for _, value, source in values if source == "text_layer"}
        if len(trusted_values) > 1 and field in {"applicant_name", "sponsor_id"}:
            conflicts.add("sponsor_mismatch" if field == "sponsor_id" else "identity_conflict")
    return conflicts


def ocr_spans(pdf: Path, scratch: Path, pages_to_read: set[int] | None = None) -> list[Span]:
    page_paths = render_pdf(pdf, scratch / pdf.stem, dpi=160)
    output: list[Span] = []
    doc = fitz.open(pdf)
    for page_index, page_path in enumerate(page_paths, start=1):
        if pages_to_read is not None and page_index not in pages_to_read:
            continue
        # The original page is evidence-preserving; contrast/threshold retries
        # are reserved for a future low-confidence fallback so they cannot erase
        # faint risk text or create conflicts with a clean read.
        original = variants(page_path)[0][1]
        original_lines = read_lines(original)
        for line in original_lines:
            if line.confidence >= 0.25:
                output.append(Span(line.text, page_index, line.x0, line.y0, line.x1, line.y1, 9, 0, "ocr:original"))
        # A contrast retry is used only to surface faint risk wording.  Its
        # field candidates cannot outrank visible text, so it cannot displace
        # clean transcription from the original page.
        risk_words = ("biohazard", "warrant", "tampering", "embargo", "rescind", "illegible")
        if not any(any(word in line.text.casefold() for word in risk_words) for line in original_lines):
            contrast = variants(page_path)[1][1]
            for line in read_lines(contrast):
                if line.confidence >= 0.35 and any(word in line.text.casefold() for word in risk_words):
                    output.append(Span(line.text, page_index, line.x0, line.y0, line.x1, line.y1, 9, 0, "ocr:contrast_risk"))
        page_text = " ".join(line.text for line in original_lines).casefold()
        receipt_hint = bool(re.search(r"fe[eag]\s+receipt|fee\s+stat", page_text))
        status_readable = any(canonicalize_fee_status(line.text) for line in original_lines)
        # Receipts live in the upper band of full-page scans.  Crop when the
        # title is visible but the status token is missing/mangled, on any page.
        if receipt_hint and not status_readable:
            width, height = original.size
            fee_crop = original.crop((int(width * 0.02), int(height * 0.05), int(width * 0.92), int(height * 0.42)))
            fee_crop = fee_crop.resize((fee_crop.width * 2, fee_crop.height * 2))
            for line in read_lines(fee_crop):
                if line.confidence >= 0.25:
                    output.append(Span(line.text, page_index, line.x0, line.y0, line.x1, line.y1, 9, 0, "ocr:fee_crop"))
        # Prefer OCR of the embedded scan raster (often sharper than a 160dpi
        # page render).  Skip small portraits; keep letter-sized receipt pages.
        page = doc[page_index - 1]
        for imginfo in page.get_images(full=True):
            xref = imginfo[0]
            try:
                pix = fitz.Pixmap(doc, xref)
            except Exception:
                continue
            if pix.n >= 5:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            if pix.width < 800 or pix.height < 800:
                continue
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            width, height = image.size
            header = image.crop((0, 0, max(1, int(width * 0.88)), max(1, int(height * 0.42))))
            header = ImageOps.autocontrast(ImageOps.grayscale(header))
            for line in read_lines(header):
                if line.confidence >= 0.25:
                    output.append(Span(line.text, page_index, line.x0, line.y0, line.x1, line.y1, 9, 0, "ocr:embedded_header"))
            # Sparse layout + speckled B-13 scans: PSM 11 on a 2x header crop
            # recovers ``Observed flags: bichazard_red`` where PSM 6 returns
            # only the form title.
            sparse = header.resize((header.width * 2, header.height * 2), Image.Resampling.LANCZOS)
            sparse_text = read_text(sparse, psm=11)
            if sparse_text.strip():
                output.append(Span(sparse_text.strip(), page_index, 0, 0, 1, 1, 9, 0, "ocr:embedded_psm11"))
    doc.close()
    return output


def predict_pdf(pdf: Path, scratch: Path, use_ocr: bool = True) -> dict[str, object]:
    record = blank_record(pdf.stem)
    spans = trusted_spans(pdf)
    # OCR only packet pages without meaningful visible PDF spans.  The generic
    # challenge footer is not document evidence: scan-only risk pages otherwise
    # contain only that footer and must still be OCR'd.
    generic_footer = "synthetic hiring challenge document"
    meaningful_pages = {
        span.page for span in spans
        if span.size >= 8
        and not span.text.startswith("Packet ")
        and normalized(span.text) != generic_footer
    }
    if use_ocr:
        doc = fitz.open(pdf)
        missing_pages = set(range(1, len(doc) + 1)) - meaningful_pages
        if missing_pages:
            spans.extend(ocr_spans(pdf, scratch, missing_pages))
    candidates, flags, manual = candidate_values(spans)
    field_sources: dict[str, str] = {}
    flags.update(pick_fields(record, candidates, field_sources))
    record["risk_flags"] = normalize_flags(flags)
    normalize_record(record)

    if manual:
        record["adjudication"] = manual
        record["confidence"] = 0.94
        return record

    policy = apply_safety_policy(record)
    world = str(record["home_world"]).casefold()
    if world in EMBARGO_WORLDS or (world == CONDITIONAL_EMBARGO_WORLD and record["visa_class"] != "DIP-1"):
        policy_decision = "DENIED"
    elif record["sponsor_id"] in REVOKED_SPONSORS and record["visa_class"] != "DIP-1":
        policy_decision = "DENIED"
    elif set(str(record["risk_flags"]).split("|")) & HARD_FLAGS:
        policy_decision = "DENIED"
    elif policy.decision:
        policy_decision = policy.decision
    else:
        policy_decision = "APPROVED"
    # Sponsor-attestation prose is useful transcription evidence, but it is a
    # lower-precedence supporting document.  It cannot on its own turn a case
    # that lacked a trusted sponsor field into an automatic approval.
    if policy_decision == "APPROVED" and any(
        field_sources.get(field) == "sponsor_attestation" for field in ("sponsor_id", "visa_class")
    ):
        policy_decision = "NEEDS_REVIEW"
    record["adjudication"], record["confidence"] = policy_decision, CONFIDENCE_BY_DECISION[policy_decision]
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="MIB classical visible-evidence baseline")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--start", type=int, default=0, help="zero-based packet offset")
    parser.add_argument("--limit", type=int, default=None, help="maximum packets to process")
    parser.add_argument("--append", action="store_true", help="append JSONL rather than replacing it")
    parser.add_argument("--scratch", type=Path, default=Path("/tmp/mib-classical"))
    parser.add_argument("--case-list", type=Path, help="optional newline-delimited case IDs to process")
    args = parser.parse_args()
    scratch = args.scratch
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.case_list:
        case_ids = [line.strip() for line in args.case_list.open() if line.strip()]
        pdfs = [args.input_dir / f"{case_id}.pdf" for case_id in case_ids]
    else:
        pdfs = sorted(args.input_dir.glob("*.pdf"))
    pdfs = pdfs[args.start:]
    if args.limit is not None:
        pdfs = pdfs[:args.limit]
    with args.output_path.open("a" if args.append else "w") as handle:
        for index, pdf in enumerate(pdfs, start=1):
            handle.write(json.dumps(predict_pdf(pdf, scratch, use_ocr=not args.no_ocr), sort_keys=True) + "\n")
            handle.flush()
            if index % 100 == 0:
                print(f"Predicted {index} packets")


if __name__ == "__main__":
    main()
