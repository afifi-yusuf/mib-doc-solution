# MIB Doc Challenge: Earning Approvals from Trusted Evidence

Submission for the [MIB Doc Challenge](https://github.com/8090-inc/mib-doc-challenge).

I come from a computer vision and deep learning background and started with packet CNNs, stamp and region heads, autoencoder-style denoising, and VLM probes. Those experiments lost to a simpler truth: this leaderboard is decided by catastrophic false approvals, so I shipped a multi-page evidence pipeline that earns every approval from trusted evidence. It combines render and OCR fallbacks, typed field evidence, layout proofs for CFA-safe reclaim, and field-manual adjudication — fully offline.

**Public train score (official `evaluate.py`): 120.14 / 150**, from 64.09/80 classification, 40.84/50 extraction, and 15.20/20 calibration, with zero missing cases and **zero catastrophic false approvals**. Runtime stays inside the 6-second-per-PDF budget on 4 vCPUs.

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
2. Collect → resolve → Rapid/forensic gap-fill → layout field repairs → decide → layout unlocks → emit.
3. Fields are typed `FieldEvidence` (`RESOLVED` / `UNKNOWN` / `CONTESTED`); schema `risk_flags=none` is an emit fallback, not clearance. APPROVED needs resolved risk or a trusted text-layer Finding.
4. OCR scan-only and image-heavy pages; RapidOCR fills UNKNOWN fee/risk/visa only (never overrides `text_layer`).
5. Layout repairs fill weak/unknown fee, name, visa, arrival, sponsor, purpose, home world, and species from injection-stripped visible text (no SYSTEM / answer-key overlays).
6. After fail-closed `decide`, two CFA-safe unlocks may promote `NEEDS_REVIEW` → `APPROVED`:
   - **Clean packet:** text-layer risk clearance (`Observed flags: none`) + layout-agreeing fee + real arrival — never OCR-only clearance.
   - **Layout consensus:** DIP-1 / XW-2 only, serialized `paid`, visible `Amount $809`, unique registry↔applicant match, no layout risk tokens (XW-1 excluded).
7. Calibrate confidence by decision class / unlock path.

Details and failure modes: [`MEMO.md`](MEMO.md).

## Layout

| Path | Role |
| --- | --- |
| `src/mib_solution/classical.py` | Packet extraction + staged predict |
| `src/mib_solution/evidence.py` | FieldEvidence / FieldState types |
| `src/mib_solution/layout_proofs.py` | Injection-stripped layout proofs (`$809`, names, risk vetoes) |
| `src/mib_solution/layout_gates.py` | Field repairs + clean-packet / consensus unlocks |
| `src/mib_solution/policy.py` | Field-manual safety rules |
| `src/mib_solution/extract.py` | Field patterns and label parsing |
| `src/mib_solution/render.py` | Page rendering for OCR |
| `src/mib_solution/ocr.py` | Tesseract wrapper |
| `src/mib_solution/rapid_fill.py` | RapidOCR UNKNOWN-field fill |
| `src/mib_solution/forensic_risk.py` | Scan-slip risk clearance / flags |
| `src/mib_solution/contracts.py` | Prediction record schema and normalization |
| `src/mib_solution/infer.py` | Docker / CLI entrypoint |
| `Dockerfile` / `run.sh` | Contest image |
| `tests/` | Unit tests |
