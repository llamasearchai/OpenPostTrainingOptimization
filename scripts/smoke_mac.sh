#!/usr/bin/env bash
set -euo pipefail

opt status --device auto
opt quantize -m gpt2 --method int8 -o outputs/q --device mps
opt sparsify -m gpt2 -s 0.5 --pattern 2:4 --device mps -o outputs/sparsity
opt profile -m gpt2 --profile latency --device mps -e outputs/profile.json
