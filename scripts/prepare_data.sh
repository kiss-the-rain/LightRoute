#!/usr/bin/env bash
set -euo pipefail

python3 src/main.py --config configs/base.yaml --mode prepare_data "$@"
