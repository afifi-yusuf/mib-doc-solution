#!/usr/bin/env sh
set -eu

archive="${1:-mib-doc-challenge-public-data-v2026-07-07.zip}"
expected="a9bb8c1bbf51346ebf49c2e3e1acdb7a5d6cd0760162767b0d133c7b7200f3c4"

if ! command -v hf >/dev/null 2>&1; then
  echo "Install Hugging Face CLI, then rerun this script." >&2
  exit 1
fi
hf download arjun-krishna1/mib-doc-challenge-data "$archive" --repo-type dataset --local-dir .
actual="$(shasum -a 256 "$archive" | awk '{print $1}')"
test "$actual" = "$expected"
unzip -q "$archive"

