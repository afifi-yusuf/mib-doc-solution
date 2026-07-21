"""Small offline classifier over *visible* document text.

This is deliberately not a language model: it is a character n-gram TF-IDF
representation and multinomial logistic regression trained only on the public
packets.  It complements deterministic policy rules on clean text-layer PDFs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .classical import trusted_spans

CLASSES = ("APPROVED", "DENIED", "NEEDS_REVIEW")


def visible_text(pdf: Path) -> str:
    """Join only trusted visible spans in reading order; never consume hidden PDF text."""
    spans = sorted(trusted_spans(pdf), key=lambda span: (span.page, round(span.y0 / 3), span.x0))
    return "\n".join(span.text for span in spans)


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=2,
        max_features=80_000, sublinear_tf=True, strip_accents="unicode",
    )


def make_classifier() -> LogisticRegression:
    return LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", multi_class="multinomial")


def read_rows(labels_path: Path) -> list[dict[str, str]]:
    with labels_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_training_texts(pdf_dir: Path, labels_path: Path, text_cache: Path | None = None) -> tuple[list[dict[str, str]], list[str]]:
    rows = read_rows(labels_path)
    cached: dict[str, str] = {}
    if text_cache and text_cache.is_file():
        with text_cache.open() as handle:
            cached = {item["case_id"]: item["text"] for item in map(json.loads, handle)}
    texts: list[str] = []
    for index, row in enumerate(rows, start=1):
        text = cached.get(row["case_id"])
        if text is None:
            text = visible_text(pdf_dir / f"{row['case_id']}.pdf")
        texts.append(text)
        if index % 100 == 0:
            print(f"Read visible text {index}/{len(rows)}")
    return rows, texts


def oof_probabilities(texts: list[str], labels: list[str], folds: int) -> list[dict[str, float]]:
    result: list[dict[str, float] | None] = [None] * len(texts)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=8090)
    for fold, (train_index, test_index) in enumerate(splitter.split(texts, labels), start=1):
        vectorizer = make_vectorizer()
        matrix = vectorizer.fit_transform([texts[index] for index in train_index])
        classifier = make_classifier().fit(matrix, [labels[index] for index in train_index])
        probabilities = classifier.predict_proba(vectorizer.transform([texts[index] for index in test_index]))
        for index, values in zip(test_index, probabilities):
            result[index] = {label: float(value) for label, value in zip(classifier.classes_, values)}
        print(f"Completed text OOF fold {fold}/{folds}")
    return [value or {} for value in result]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a visible-text MIB classifier")
    parser.add_argument("--train-pdfs", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--oof", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--text-cache", type=Path, help="reusable JSONL visible-text cache")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    if args.extract_only:
        if not args.text_cache:
            parser.error("--extract-only requires --text-cache")
        rows = read_rows(args.labels)[args.start:]
        if args.limit is not None:
            rows = rows[:args.limit]
        args.text_cache.parent.mkdir(parents=True, exist_ok=True)
        with args.text_cache.open("a" if args.append else "w") as handle:
            for index, row in enumerate(rows, start=1):
                handle.write(json.dumps({"case_id": row["case_id"], "text": visible_text(args.train_pdfs / f"{row['case_id']}.pdf")}) + "\n")
                if index % 50 == 0:
                    print(f"Cached visible text {index}/{len(rows)}")
        return

    rows, texts = load_training_texts(args.train_pdfs, args.labels, args.text_cache)
    labels = [row["adjudication"] for row in rows]
    if args.oof:
        probabilities = oof_probabilities(texts, labels, args.folds)
        args.oof.parent.mkdir(parents=True, exist_ok=True)
        with args.oof.open("w") as handle:
            for row, probability in zip(rows, probabilities):
                handle.write(json.dumps({"case_id": row["case_id"], "probabilities": probability}, sort_keys=True) + "\n")

    vectorizer = make_vectorizer()
    classifier = make_classifier().fit(vectorizer.fit_transform(texts), labels)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "classifier": classifier}, args.artifact, compress=3)
    print(f"Saved visible-text classifier to {args.artifact}")


if __name__ == "__main__":
    main()
