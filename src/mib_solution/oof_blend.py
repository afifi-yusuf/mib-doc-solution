"""Evaluate decision blending from saved out-of-fold probabilities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--respect-base-denials", action="store_true")
    args = parser.parse_args()
    probabilities = {row["case_id"]: row["probabilities"] for row in map(json.loads, args.oof.open())}
    with args.output.open("w") as handle:
        for record in map(json.loads, args.base.open()):
            probability = probabilities[record["case_id"]]
            decision, confidence = max(probability.items(), key=lambda item: item[1])
            if confidence < args.min_confidence:
                decision = "NEEDS_REVIEW"
            if args.respect_base_denials and record["adjudication"] == "DENIED":
                decision = "DENIED"
                confidence = max(confidence, float(record["confidence"]))
            record["adjudication"] = decision
            record["confidence"] = round(float(confidence), 4)
            handle.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
