#!/usr/bin/env bash
#
# Speculative decoding baseline for GSM8K using a single-process accelerate launch.
#
# Usage:
#   bash scripts/run_gsm8k_sd.sh /absolute/path/to/gsm8k
#
# Environment requirements:
#   * Draft and target model aliases must be defined in src/util.py:model_zoo.
#   * A single GPU (default CUDA device 0) is sufficient for sd mode.
#
# Optional overrides via environment variables:
#   EXP_NAME     - Experiment directory under ./exp/ (default: GSM8K_SD_llama2)
#   DRAFT_ALIAS  - Draft model alias (default: llama-2-7b)
#   TARGET_ALIAS - Target model alias (default: llama-2-70b)
#   GAMMA        - Speculative steps per round (default: 4)
#   NUM_SAMPLES  - Completions per question (default: 1)
#   MAX_TOKENS   - Generation budget (default: 256)
#   TEMPERATURE  - Sampling temperature (default: 0.2)

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/gsm8k" >&2
  exit 1
fi

DATA_PATH="$1"
EXP_NAME="${EXP_NAME:-GSM8K_SD_llama2}"
DRAFT_ALIAS="${DRAFT_ALIAS:-llama-2-7b}"
TARGET_ALIAS="${TARGET_ALIAS:-llama-2-70b}"
GAMMA="${GAMMA:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.2}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
accelerate launch --num_processes 1 benchmark/eval_gsm8k.py \
  --data_path "${DATA_PATH}" \
  --eval_mode sd \
  --gamma "${GAMMA}" \
  -n "${NUM_SAMPLES}" \
  -e "${EXP_NAME}" \
  --draft_model "${DRAFT_ALIAS}" \
  --target_model "${TARGET_ALIAS}" \
  --max_tokens "${MAX_TOKENS}" \
  --temp "${TEMPERATURE}"
