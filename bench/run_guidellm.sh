#!/usr/bin/env bash
# Cross-check bench.py numbers with GuideLLM's sweep (TTFT/ITL distributions).
# Usage: IQLAB_API_KEY=... ./bench/run_guidellm.sh https://api.cerebras.ai/v1 llama3.1-8b
set -euo pipefail
TARGET=${1:?base url}; MODEL=${2:?model}
OUT=results/guidellm_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
OPENAI_API_KEY="${IQLAB_API_KEY:-}" guidellm benchmark \
  --target "$TARGET" --model "$MODEL" \
  --rate-type sweep --max-seconds 30 \
  --data "prompt_tokens=256,output_tokens=128" \
  --output-path "$OUT/benchmarks.json"
echo "results in $OUT"
