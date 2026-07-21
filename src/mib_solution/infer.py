"""Docker entrypoint: parallel offline packet predictor."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .classical import predict_pdf


def _worker(pdf_path: str, scratch_root: str) -> dict[str, object]:
    pdf = Path(pdf_path)
    scratch = Path(scratch_root) / pdf.stem
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        return predict_pdf(pdf, scratch, use_ocr=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline MIB packet predictor")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("MIB_WORKERS", "4")),
        help="parallel PDF workers (default 4, matching scoring CPUs)",
    )
    args = parser.parse_args()

    pdfs = sorted(args.input_dir.glob("*.pdf"))
    scratch_root = Path("/tmp/mib-infer")
    shutil.rmtree(scratch_root, ignore_errors=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    workers = max(1, args.workers)
    results: dict[str, dict[str, object]] = {}
    total = len(pdfs)
    print(f"cases={total} workers={workers}", flush=True)
    done = 0
    if workers == 1:
        for pdf in pdfs:
            record = _worker(str(pdf), str(scratch_root))
            results[str(record["case_id"])] = record
            done += 1
            if done % 50 == 0 or done == total:
                print(f"{done}/{total}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_worker, str(pdf), str(scratch_root)): pdf.stem for pdf in pdfs
            }
            for future in as_completed(futures):
                record = future.result()
                results[str(record["case_id"])] = record
                done += 1
                if done % 50 == 0 or done == total:
                    print(f"{done}/{total}", flush=True)

    with args.output_path.open("w") as handle:
        for pdf in pdfs:
            handle.write(json.dumps(results[pdf.stem], sort_keys=True) + "\n")
    print(f"wrote {args.output_path} n={total}", flush=True)


if __name__ == "__main__":
    main()
