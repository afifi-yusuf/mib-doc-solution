# MIB Vision-First Submission

An offline, CPU-only submission for the MIB Doc Challenge. It uses rendered PDF pages as the evidence source, OCR for exact field transcription, deterministic safety rules, and an optional compact packet CNN trained only on the public training set.

## Compliance

The runtime contains no LLM, VLM, multimodal foundation model, cloud OCR, network client, or API key. All OCR is local Tesseract; the learned model is a small custom PyTorch CNN trained from scratch.

## Train

Download the public archive into a local `data/` directory, then run:

```bash
PYTHONPATH=src python -m mib_solution.train \
  --train-pdfs data/train \
  --labels data/train_labels.csv \
  --artifacts artifacts \
  --cache artifacts/render-cache
```

Training performs stratified five-fold out-of-fold evaluation, writes calibrated thresholds and label maps, then trains final seeds. `artifacts/model.pt` and `artifacts/label_maps.json` are required by the Docker image.

## Build and run

```bash
docker build -t mib-vision .
docker run --rm --network none \
  --mount type=bind,src="$PWD/data/validation",dst=/input,readonly \
  --mount type=bind,src="$PWD/out",dst=/output \
  mib-vision /input /output/predictions.jsonl
```

The final image must be tested with the challenge's `scripts/run_docker_submission.py` and `scripts/validate_submission.py` before submission.
