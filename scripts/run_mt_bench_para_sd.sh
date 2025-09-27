#!/usr/bin/env bash
#
# Parallel speculative decoding run over MT-Bench with accelerate.
#
# Usage:
#   bash scripts/run_mt_bench_para_sd.sh /absolute/path/to/mt-bench
#
# Notes:
#   * Ensure src/util.py:model_zoo maps the alias models used below to local checkpoints.
#   * fastchat must be installed (provides the conversation templates consumed by MT-Bench).
#   * Two GPUs (IDs 0 and 1 by default) are required for para_sd mode.
#
# Optional overrides via environment variables:
#   EXP_NAME     - Experiment directory under ./exp/ (default: MT_PSD_llama2)
#   DRAFT_ALIAS  - Draft model alias (default: llama-2-7b)
#   TARGET_ALIAS - Target model alias (default: llama-2-70b)
#   GAMMA        - Speculative steps per round (default: 4)
#   NUM_SAMPLES  - Completions per prompt (default: 1)
#   MAX_TOKENS   - Generation budget (default: 512)
#   TEMPERATURE  - Sampling temperature (default: 0.7)

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/mt-bench" >&2
  exit 1
fi

DATA_PATH="$1"
EXP_NAME="${EXP_NAME:-MT_PSD_llama2}"
DRAFT_ALIAS="${DRAFT_ALIAS:-llama-2-7b}"
TARGET_ALIAS="${TARGET_ALIAS:-llama-2-70b}"
GAMMA="${GAMMA:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.7}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
accelerate launch --num_processes 2 benchmark/eval_mt_bench.py \
  --data_path "${DATA_PATH}" \
  --eval_mode para_sd \
  --gamma "${GAMMA}" \
  -n "${NUM_SAMPLES}" \
  -e "${EXP_NAME}" \
  --draft_model "${DRAFT_ALIAS}" \
  --target_model "${TARGET_ALIAS}" \
  --max_tokens "${MAX_TOKENS}" \
  --temp "${TEMPERATURE}"
