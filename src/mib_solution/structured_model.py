"""A compact supervised policy-exception model over recovered packet fields."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import StratifiedKFold


def features(record: dict[str, object]) -> dict[str, object]:
    visa = str(record.get("visa_class", "unknown"))
    sponsor = str(record.get("sponsor_id", "SPN-0000"))
    world = str(record.get("home_world", "unknown")).casefold()
    fee = str(record.get("fee_status", "unknown"))
    flags = str(record.get("risk_flags", "none"))
    return {
        "visa": visa,
        "fee": fee,
        "world": world,
        "purpose": str(record.get("declared_purpose", "unknown")).casefold(),
        "flags": flags,
        "sponsor": sponsor,
        "diplomatic": visa == "DIP-1",
        "transit": visa == "TRANSIT-7",
        "unpaid": fee == "unpaid",
        "unknown_fee": fee == "unknown",
        "missing_sponsor": sponsor == "SPN-0000",
        "manual_revoked_sponsor": sponsor in {"SPN-0007", "SPN-0139", "SPN-4040"},
        "inferred_revoked_sponsor": sponsor in {"SPN-9090", "SPN-7331", "SPN-2718"},
        "embargo_world": world in {"trappist-1e", "eris relay"},
        "review_flag": flags != "none" and not any(flag in flags for flag in ("memory_tampering", "planetary_embargo", "active_warrant", "biohazard_red")),
    }


def classifier() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=500, max_depth=10, min_samples_leaf=2,
        class_weight="balanced", max_features=0.8, n_jobs=-1, random_state=8090,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a compact structured MIB policy model")
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--oof", type=Path)
    args = parser.parse_args()
    records = {row["case_id"]: row for row in map(json.loads, args.records.open())}
    with args.labels.open(newline="") as handle:
        labels = list(csv.DictReader(handle))
    x = [features(records[row["case_id"]]) for row in labels]
    y = [row["adjudication"] for row in labels]
    if args.oof:
        output: list[dict[str, float] | None] = [None] * len(y)
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=8090)
        for fold, (train, valid) in enumerate(folds.split(x, y), start=1):
            vectorizer = DictVectorizer(sparse=True)
            train_matrix = vectorizer.fit_transform([x[index] for index in train])
            model = classifier().fit(train_matrix, [y[index] for index in train])
            probabilities = model.predict_proba(vectorizer.transform([x[index] for index in valid]))
            for index, probability in zip(valid, probabilities):
                output[index] = {label: float(value) for label, value in zip(model.classes_, probability)}
            print(f"Completed structured OOF fold {fold}/5")
        with args.oof.open("w") as handle:
            for row, probability in zip(labels, output):
                handle.write(json.dumps({"case_id": row["case_id"], "probabilities": probability}, sort_keys=True) + "\n")
    vectorizer = DictVectorizer(sparse=True)
    model = classifier().fit(vectorizer.fit_transform(x), y)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "model": model}, args.artifact, compress=3)
    print(f"Saved structured policy model to {args.artifact}")


if __name__ == "__main__":
    main()
