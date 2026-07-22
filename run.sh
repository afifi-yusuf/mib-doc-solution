#!/usr/bin/env bash
set -euo pipefail

input_dir="${1:?usage: ./run.sh <input_pdf_dir> <output_path>}"
output_path="${2:?usage: ./run.sh <input_pdf_dir> <output_path>}"

python3 -m mib_solution.infer "$input_dir" "$output_path"
