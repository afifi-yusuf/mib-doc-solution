from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from .contracts import blank_record, is_iso_date
from .extract import extract
from .ocr import OCRPage, read_tsv
from .policy import apply_safety_policy
from .render import render_pdf, variants
from .vision_io import load_predictor


def normalize_fields(record: dict[str, object]) -> None:
    fee = str(record.get("fee_status", "unknown")).casefold()
    record["fee_status"] = next((item for item in ("paid", "waived", "unpaid", "unknown") if item in fee), "unknown")
    date = str(record.get("arrival_date", ""))
    if not is_iso_date(date):
        record["arrival_date"] = "1900-01-01"
    sponsor = str(record.get("sponsor_id", ""))
    if not sponsor.startswith("SPN-") or len(sponsor) != 8 or not sponsor[4:].isdigit():
        record["sponsor_id"] = "SPN-0000"


def ocr_packet(page_paths: list[Path]) -> list[OCRPage]:
    pages: list[OCRPage] = []
    for index, page_path in enumerate(page_paths, start=1):
        for variant_name, image in variants(page_path):
            text, confidence = read_tsv(image)
            if text:
                pages.append(OCRPage(index, variant_name, text, confidence))
    return pages


def predict_case(pdf_path: Path, scratch: Path, predictor) -> dict[str, object]:
    case_id = pdf_path.stem
    case_dir = scratch / case_id
    page_paths = render_pdf(pdf_path, case_dir)
    record = blank_record(case_id)
    extracted, _candidates = extract(ocr_packet(page_paths))
    record.update(extracted)
    normalize_fields(record)

    if predictor:
        visual, confidence = predictor.predict(page_paths)
        # OCR is authoritative for exact fields; visual heads only backfill meaningful classes.
        for field in ("visa_class", "fee_status"):
            if record.get(field) in {"unknown", "SPN-0000"} and visual.get(field):
                record[field] = visual[field]
        decision = visual.get("adjudication", "NEEDS_REVIEW")
        if confidence < predictor.threshold:
            decision = "NEEDS_REVIEW"
        record["adjudication"] = decision
        record["confidence"] = round(confidence, 4)
    else:
        record["adjudication"] = "NEEDS_REVIEW"
        record["confidence"] = 0.01

    # A trusted policy violation is a safety veto over the learned primary signal.
    policy = apply_safety_policy(record)
    if policy.decision:
        record["adjudication"] = policy.decision
        record["confidence"] = max(float(record["confidence"]), 0.9 if policy.decision == "DENIED" else 0.55)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline MIB vision-first predictor")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    model_path = Path(os.environ.get("MIB_MODEL_PATH", "/app/artifacts/model.pt"))
    maps_path = Path(os.environ.get("MIB_LABEL_MAPS", "/app/artifacts/label_maps.json"))
    predictor = load_predictor(model_path, maps_path)
    scratch = Path("/tmp/mib-render")
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w") as handle:
        for pdf_path in sorted(args.input_dir.glob("*.pdf")):
            handle.write(json.dumps(predict_case(pdf_path, scratch, predictor), sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
