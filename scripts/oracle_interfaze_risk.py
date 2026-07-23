#!/usr/bin/env python3
"""Local-only Interfaze OCR oracle for fail-closed reclaim analysis.

Reads INTERFAZE_API_KEY from the environment (or mib-doc-challenge/.env).
Never imported by mib_solution runtime / Docker.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import urllib.request
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
CHALLENGE_ENV = Path("/Users/yusufafifi/Desktop/mib-doc-challenge/.env")


def load_key() -> str:
    key = os.environ.get("INTERFAZE_API_KEY", "").strip()
    if key:
        return key
    if CHALLENGE_ENV.exists():
        for line in CHALLENGE_ENV.read_text().splitlines():
            if line.startswith("INTERFAZE_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("INTERFAZE_API_KEY not set")


def interfaze_ocr(key: str, png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode()
    payload = {
        "model": "interfaze-beta",
        "messages": [
            {"role": "system", "content": "<task>ocr</task>"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            },
        ],
    }
    req = urllib.request.Request(
        "https://api.interfaze.ai/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode())
    content = body["choices"][0]["message"].get("content") or ""
    try:
        parsed = json.loads(content)
        result = parsed.get("result") or {}
        return str(result.get("extracted_text") or content)
    except json.JSONDecodeError:
        return content


def risk_signals(text: str) -> dict[str, object]:
    folded = text.casefold()
    cleared = bool(
        re.search(r"observed flags?\s*:\s*none\b", text, re.I)
        or re.search(r"\brisk flags?\s*:\s*none\b", text, re.I)
    )
    hard = sorted(
        {
            name
            for name, pat in (
                ("active_warrant", r"active\s*warrant"),
                ("biohazard_red", r"biohazard|bichazard"),
                ("memory_tampering", r"memory\s*tamper"),
                ("illegible_biometrics", r"illegible\s*biometric|b-13"),
            )
            if re.search(pat, folded)
        }
    )
    return {
        "cleared": cleared,
        "has_b13": "b-13" in folded or "biometric" in folded,
        "has_registry_clear": bool(re.search(r"\bclear\b", folded)),
        "hard_flag_hints": hard,
        "has_sample_denial": "sample denial" in folded,
        "has_finding": bool(re.search(r"\bfinding\s*:", folded)),
    }


def pure_miss_ids(preds_path: Path, labels_path: Path) -> list[str]:
    truth = {r["case_id"]: r for r in csv.DictReader(labels_path.open())}
    preds = {json.loads(l)["case_id"]: json.loads(l) for l in preds_path.open()}
    out = []
    keys = ["risk_flags", "fee_status", "visa_class", "sponsor_id", "species_code", "home_world"]
    for cid, t in truth.items():
        p = preds.get(cid)
        if not p:
            continue
        if t["adjudication"] != "APPROVED" or p["adjudication"] != "NEEDS_REVIEW":
            continue
        if all(str(p.get(k, "")).lower() == str(t.get(k, "")).lower() for k in keys):
            out.append(cid)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument(
        "--preds",
        type=Path,
        default=Path("/tmp/mib_fieldevidence_preds2.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/mib_interfaze_oracle.jsonl"),
    )
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    key = load_key()
    ids = args.case or pure_miss_ids(args.preds, ROOT / "data" / "train_labels.csv")
    ids = ids[: args.limit]
    print(f"oracle cases={len(ids)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for index, cid in enumerate(ids, start=1):
            pdf = ROOT / "data" / "train" / f"{cid}.pdf"
            doc = fitz.open(pdf)
            pages = []
            for page_index, page in enumerate(doc, start=1):
                n_img = len(page.get_images(full=True))
                # Always oracle image pages; also first page if nothing else.
                if n_img == 0 and page_index > 1:
                    continue
                pix = page.get_pixmap(dpi=args.dpi, alpha=False)
                text = interfaze_ocr(key, pix.tobytes("png"))
                pages.append(
                    {
                        "page": page_index,
                        "n_images": n_img,
                        "text": text,
                        "signals": risk_signals(text),
                    }
                )
            doc.close()
            any_cleared = any(p["signals"]["cleared"] for p in pages)
            any_b13 = any(p["signals"]["has_b13"] for p in pages)
            row = {
                "case_id": cid,
                "any_cleared": any_cleared,
                "any_b13": any_b13,
                "pages": pages,
            }
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(
                f"{index}/{len(ids)} {cid} cleared={any_cleared} b13={any_b13} pages={len(pages)}",
                flush=True,
            )
    # Summary
    rows = [json.loads(l) for l in args.out.open()]
    print(
        "summary",
        {
            "n": len(rows),
            "cleared": sum(1 for r in rows if r["any_cleared"]),
            "b13": sum(1 for r in rows if r["any_b13"]),
            "neither": sum(1 for r in rows if not r["any_cleared"] and not r["any_b13"]),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
