#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash radio.sh <radio_download_dir>" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
download_dir="$1"

manifest_path="$(
  python "$script_dir/build_radio_manifest.py" \
    --metadata_csv "$script_dir/nsclc_radiogenomics.csv" \
    --output_dir "$download_dir"
)"

cd "$download_dir"
s5cmd --no-sign-request --endpoint-url https://s3.amazonaws.com run "$manifest_path"
