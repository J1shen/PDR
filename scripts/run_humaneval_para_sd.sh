#!/usr/bin/env bash
#
# Parallel speculative decoding run over HumanEval using DeepSeek draft/target checkpoints.
#
# Usage:
#   bash scripts/run_humaneval_para_sd.sh /absolute/path/to/humaneval
#
# Required environment:
#   * Update src/util.py:model_zoo with the local paths for the deepseek checkpoints.
#   * Provide a CUDA environment with two visible GPUs (IDs 0 and 1 by default).
#
# Optional overrides via environment variables:
#   EXP_NAME        - Name of the experiment folder under ./exp/ (default: H_PSD_deepseek1367b)
#   DRAFT_ALIAS     - Draft model alias defined in src/util.py (default: deepseek-1.3b)
#   TARGET_ALIAS    - Target model alias defined in src/util.py (default: deepseek-6.7b)
#   MAX_TOKENS      - Generation budget (default: 512)
#   TEMPERATURE     - Sampling temperature (default: 0.0)
#   GAMMA           - Number of speculative steps per draft round (default: 4)
#   NUM_SAMPLES     - Completions per HumanEval prompt (default: 1)
#
# The script simply mirrors the accelerate command; adjust flags as needed.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/humaneval" >&2
  exit 1
fi

DATA_PATH="$1"
EXP_NAME="${EXP_NAME:-H_PSD_deepseek1367b}"
DRAFT_ALIAS="${DRAFT_ALIAS:-deepseek-1.3b}"
TARGET_ALIAS="${TARGET_ALIAS:-deepseek-6.7b}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
GAMMA="${GAMMA:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
accelerate launch --num_processes 2 benchmark/eval_humaneval.py \
  --data_path "${DATA_PATH}" \
  --eval_mode para_sd \
  --gamma "${GAMMA}" \
  -n "${NUM_SAMPLES}" \
  -e "${EXP_NAME}" \
  --draft_model "${DRAFT_ALIAS}" \
  --target_model "${TARGET_ALIAS}" \
  --max_tokens "${MAX_TOKENS}" \
  --temp "${TEMPERATURE}"
