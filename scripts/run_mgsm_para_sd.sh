#!/usr/bin/env bash
#
# Parallel speculative decoding run for MGSM multilingual math benchmark.
#
# Usage:
#   bash scripts/run_mgsm_para_sd.sh /absolute/path/to/mgsm
#
# Requirements:
#   * Two GPUs (IDs 0 and 1 by default) for para_sd mode.
#   * Draft/target aliases must resolve to actual checkpoints via src/util.py:model_zoo.
#
# Optional environment overrides:
#   EXP_NAME     - Experiment directory under ./exp/ (default: MGSM_PSD_deepseek)
#   DRAFT_ALIAS  - Draft model alias (default: deepseek-1.3b)
#   TARGET_ALIAS - Target model alias (default: deepseek-6.7b)
#   GAMMA        - Speculative steps per round (default: 4)
#   NUM_SAMPLES  - Completions per question (default: 1)
#   MAX_TOKENS   - Generation budget (default: 256)
#   TEMPERATURE  - Sampling temperature (default: 0.2)

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/mgsm" >&2
  exit 1
fi

DATA_PATH="$1"
EXP_NAME="${EXP_NAME:-MGSM_PSD_deepseek}"
DRAFT_ALIAS="${DRAFT_ALIAS:-deepseek-1.3b}"
TARGET_ALIAS="${TARGET_ALIAS:-deepseek-6.7b}"
GAMMA="${GAMMA:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.2}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
accelerate launch --num_processes 2 benchmark/eval_mgsm.py \
  --data_path "${DATA_PATH}" \
  --eval_mode para_sd \
  --gamma "${GAMMA}" \
  -n "${NUM_SAMPLES}" \
  -e "${EXP_NAME}" \
  --draft_model "${DRAFT_ALIAS}" \
  --target_model "${TARGET_ALIAS}" \
  --max_tokens "${MAX_TOKENS}" \
  --temp "${TEMPERATURE}"
