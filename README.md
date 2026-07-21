# MIB Doc Challenge — Offline Packet Intake

Offline, CPU-only submission for the [MIB Doc Challenge](https://github.com/8090-inc/mib-doc-challenge).

A full document-engineering pipeline for messy multi-page packets: render + OCR fallbacks, cross-page evidence ranking, conflict detection, adversarial hidden-text rejection, and safety-critical adjudication under the official Docker constraints (no network, no LLM/VLM runtime, no cloud OCR).

## Runtime contract

```bash
docker build -t mib-submission .
mkdir -p /tmp/mib-output
docker run --rm --network none \
  --mount type=bind,src="$PWD/data/validation",dst=/input,readonly \
  --mount type=bind,src="/tmp/mib-output",dst=/output \
  mib-submission /input /output/predictions.jsonl
```

The image accepts exactly `<input_pdf_dir> <output_predictions_path>`, writes under `/output`, and uses `/tmp` for scratch. Parallelism defaults to 4 workers (`MIB_WORKERS`).

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# system deps: tesseract-ocr, poppler-utils

PYTHONPATH=src python -m mib_solution.infer data/train /tmp/train_preds.jsonl
PYTHONPATH=src python -m unittest discover -s tests -v
```

Score against public labels from the challenge repo:

```bash
python3 /path/to/mib-doc-challenge/scripts/evaluate.py \
  --truth data/train_labels.csv \
  --submission /tmp/train_preds.jsonl
```

## Pipeline (short)

1. Filter PDF spans for real ink; discard white/tiny hidden text that can inject fields.
2. Rank multi-document evidence (intake, registry, attestation, fee) and fuse fields across pages.
3. OCR scan-only pages with targeted crops/retries; resolve near-duplicate and glued-label noise.
4. Surface identity/sponsor conflicts; gate approvals when attestation would overrule incomplete intake.
5. Apply field-manual adjudication with CFA-aware conservatism; calibrate confidence by decision class.

Details and failure modes: [`MEMO.md`](MEMO.md).

## Layout

| Path | Role |
| --- | --- |
| `src/mib_solution/classical.py` | Packet extraction + adjudication |
| `src/mib_solution/policy.py` | Field-manual safety rules |
| `src/mib_solution/infer.py` | Docker / CLI entrypoint |
| `Dockerfile` / `run.sh` | Offline image |
| `tests/` | Unit tests |
