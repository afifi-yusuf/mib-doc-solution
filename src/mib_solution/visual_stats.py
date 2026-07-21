"""Fast classical visual features for colored stamps and scan artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

FLAGS = ("memory_tampering", "active_warrant", "biohazard_red")


def region_features(image: np.ndarray) -> np.ndarray:
    red = (image[..., 0] > image[..., 1] * 1.3) & (image[..., 0] > image[..., 2] * 1.3) & (image[..., 0] > 0.35)
    blue = (image[..., 2] > image[..., 0] * 1.15) & (image[..., 2] > image[..., 1] * 1.05) & (image[..., 2] > 0.25)
    dark = image.mean(2) < 0.3
    return np.array([red.mean(), blue.mean(), dark.mean(), *image.mean((0, 1)), *image.std((0, 1))])


def page_features(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB").resize((128, 160)), dtype=np.float32) / 255.0
    # Repeated registry/biometric stamps live in a small number of template
    # zones; preserve those zones instead of diluting their ink over a page.
    height, width = image.shape[:2]
    crops = [
        image,
        image[int(height*.08):int(height*.48), int(width*.52):int(width*.95)],   # upper-right registry zone
        image[int(height*.28):int(height*.72), int(width*.50):int(width*.95)],   # right-side stamp zone
        image[int(height*.30):int(height*.72), int(width*.28):int(width*.76)],   # central manual-stamp zone
    ]
    return np.concatenate([region_features(crop) for crop in crops])


def packet_features(case_id: str, cache: Path) -> np.ndarray:
    pages = np.stack([page_features(path) for path in sorted((cache / case_id).glob("page-*.png"))])
    return np.concatenate([pages.mean(0), pages.max(0), pages.std(0)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--trees", type=int, default=120)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.labels.open()))
    x = np.stack([packet_features(row["case_id"], args.cache) for row in rows])
    output = [{flag: 0.0 for flag in FLAGS} for _ in rows]
    for flag in FLAGS:
        y = np.array([flag in row["risk_flags"].split("|") for row in rows])
        folds = StratifiedKFold(args.folds, shuffle=True, random_state=8090)
        for train, valid in folds.split(x, y):
            model = ExtraTreesClassifier(n_estimators=args.trees, max_depth=8, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=8090)
            model.fit(x[train], y[train])
            probabilities = model.predict_proba(x[valid])[:, 1]
            for index, value in zip(valid, probabilities): output[index][flag] = float(value)
    with args.oof.open("w") as handle:
        for row, probability in zip(rows, output): handle.write(json.dumps({"case_id": row["case_id"], "probabilities": probability}) + "\n")


if __name__ == "__main__":
    main()
