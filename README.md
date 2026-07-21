# MIB Doc Challenge — Classical Visible-Evidence Pipeline

Offline, CPU-only submission for the [MIB Doc Challenge](https://github.com/8090-inc/mib-doc-challenge).

The runtime recovers fields from **visible PDF text** and **local Tesseract OCR**, then applies deterministic policy from `FIELD_MANUAL.md`. No LLM, VLM, cloud OCR, network client, or API key.

Train score (public 1,000 labels): **~117.7 / 150**, with **18** catastrophic false approvals.

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

## Approach (short)

1. Prefer native visible text spans; OCR only pages without meaningful text.
2. Rank candidates by document type (intake > registry > attestation > fee).
3. Resolve OCR noise (near-duplicate names, glued label rows, fee typos).
4. Apply hard deny / review policy; demote attestation-only approvals without trusted intake text.
5. Emit fixed confidences by decision class for calibration.

Details and failure modes: [`MEMO.md`](MEMO.md).

## Layout

| Path | Role |
| --- | --- |
| `src/mib_solution/classical.py` | Extraction + adjudication |
| `src/mib_solution/policy.py` | Safety policy |
| `src/mib_solution/infer.py` | Docker / CLI entrypoint |
| `Dockerfile` / `run.sh` | Offline image |
| `tests/` | Unit tests for parsers and policy |
