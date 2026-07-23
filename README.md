# MIB Doc Challenge: Earning Approvals from Trusted Evidence

Submission for the [MIB Doc Challenge](https://github.com/8090-inc/mib-doc-challenge).

I come from a computer vision and deep learning background and started with packet CNNs, stamp and region heads, autoencoder-style denoising, and VLM probes. Those experiments lost to a simpler truth: this leaderboard is decided by catastrophic false approvals, so I shipped a multi-page evidence pipeline that earns every approval from trusted evidence. It combines render and OCR fallbacks, document-role ranking, conflict detection, hidden-text rejection, and field-manual adjudication, and it runs fully offline.

**Public train score (official `evaluate.py`): 117.55 / 150**, from 62.79/80 classification, 39.54/50 extraction, and 15.22/20 calibration, with zero missing cases and **zero catastrophic false approvals**. Runtime stays inside the 6-second-per-PDF budget on 4 vCPUs.

## Runtime contract

The public data is not checked in. Fetch it with `scripts/download_data.sh` (Hugging Face CLI required) so `data/train/` and `data/validation/` exist, then:

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

## Pipeline

1. Filter PDF spans for real ink; discard white or tiny hidden text that can inject fields.
2. Rank multi-document evidence (intake, registry, attestation, fee) and fuse fields across pages.
3. OCR scan-only pages with targeted crops and retries; resolve near-duplicate and glued-label noise.
4. Surface identity and sponsor conflicts; gate approvals when the attestation would overrule an incomplete intake form.
5. Apply field-manual adjudication, biased against catastrophic false approvals; calibrate confidence by decision class.

Details and failure modes: [`MEMO.md`](MEMO.md).

## Layout

| Path | Role |
| --- | --- |
| `src/mib_solution/classical.py` | Packet extraction + adjudication |
| `src/mib_solution/policy.py` | Field-manual safety rules |
| `src/mib_solution/extract.py` | Field patterns and label parsing |
| `src/mib_solution/render.py` | Page rendering for OCR |
| `src/mib_solution/ocr.py` | Tesseract wrapper |
| `src/mib_solution/contracts.py` | Prediction record schema and normalization |
| `src/mib_solution/infer.py` | Docker / CLI entrypoint |
| `Dockerfile` / `run.sh` | Contest image |
| `tests/` | Unit tests |
