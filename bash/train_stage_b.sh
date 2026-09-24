#!/usr/bin/env bash
#SBATCH --job-name=cg_stage_b
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/emergent_partner_grid/logs/cg_stage_b_%j.out
#SBATCH --error=/juice6/u/jshe/emergent_partner_grid/logs/cg_stage_b_%j.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
# ---------------------------------------------------------------------------
# Stage B: CoordinationGrid recurrent IPPO, single learned ego, scripted
# partner with fixed z=1, layout sampled per episode from the 210 training
# layouts. End-of-run eval on the 45 val and 45 test layouts.
#
# Config: baselines/IPPO/config/ippo_rnn_coordination_grid.yaml
# Trainer: baselines/IPPO/ippo_rnn_coordination_grid.py
#
# Usage:
#   sbatch bash/train_stage_b.sh                       # slurm submit
#   bash   bash/train_stage_b.sh                       # local run (uses GPU
#                                                        if visible via CUDA)
#   sbatch --export=ALL,SEED=2 bash/train_stage_b.sh   # override seed
# ---------------------------------------------------------------------------
set -euo pipefail

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate emergent_partner_model

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/emergent_partner_grid}"
cd "${REPO_ROOT}" || exit 1

mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/dev/train_logs"

# --- Hyperparameters (env-overridable) ---
SEED="${SEED:-1}"
NUM_SEEDS="${NUM_SEEDS:-1}"
NUM_ENVS="${NUM_ENVS:-256}"
NUM_STEPS="${NUM_STEPS:-32}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-8}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-20000000}"
LR="${LR:-5e-4}"
PARTNER_Z="${PARTNER_Z:-1.0}"
MAX_STEPS="${MAX_STEPS:-15}"
EVAL_TRIALS_PER_LAYOUT="${EVAL_TRIALS_PER_LAYOUT:-32}"
AUGMENT_SYMMETRIES="${AUGMENT_SYMMETRIES:-true}"

TRAIN_LAYOUTS="${TRAIN_LAYOUTS:-${REPO_ROOT}/dev/grids/layouts/train}"
VAL_LAYOUTS="${VAL_LAYOUTS:-${REPO_ROOT}/dev/grids/layouts/val}"
TEST_LAYOUTS="${TEST_LAYOUTS:-${REPO_ROOT}/dev/grids/layouts/test}"

# Timestamped run tag so parallel submits don't collide on the save path.
TAG="${TAG:-stage_b_z${PARTNER_Z}_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
SAVE_PARAMS_PATH="${REPO_ROOT}/dev/train_logs/${TAG}.safetensors"

# W&B: online if ENTITY/PROJECT env vars are set; disabled otherwise.
WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-coordination_grid}"

echo "=== Stage B training  (job=${SLURM_JOB_ID:-local}, tag=${TAG}) ==="
echo "REPO_ROOT      = ${REPO_ROOT}"
echo "TRAIN_LAYOUTS  = ${TRAIN_LAYOUTS}"
echo "VAL_LAYOUTS    = ${VAL_LAYOUTS}"
echo "TEST_LAYOUTS   = ${TEST_LAYOUTS}"
echo "AUGMENT_SYM    = ${AUGMENT_SYMMETRIES}"
echo "PARTNER_Z      = ${PARTNER_Z}"
echo "TOTAL_TIMESTEPS= ${TOTAL_TIMESTEPS}"
echo "NUM_ENVS       = ${NUM_ENVS}   NUM_STEPS=${NUM_STEPS}"
echo "SAVE_PARAMS    = ${SAVE_PARAMS_PATH}"
echo "WANDB_MODE     = ${WANDB_MODE}"
echo "GPU visible:"
nvidia-smi -L 2>/dev/null || echo "  (no nvidia-smi)"
echo

python -u baselines/IPPO/ippo_rnn_coordination_grid.py \
    SEED="${SEED}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    ENV_KWARGS.layouts_dir="${TRAIN_LAYOUTS}" \
    ENV_KWARGS.partner_z="${PARTNER_Z}" \
    ENV_KWARGS.max_steps="${MAX_STEPS}" \
    ENV_KWARGS.augment_symmetries="${AUGMENT_SYMMETRIES}" \
    EVAL_LAYOUTS_DIRS.val="${VAL_LAYOUTS}" \
    EVAL_LAYOUTS_DIRS.test="${TEST_LAYOUTS}" \
    EVAL_TRIALS_PER_LAYOUT="${EVAL_TRIALS_PER_LAYOUT}" \
    SAVE_PARAMS_PATH="${SAVE_PARAMS_PATH}" \
    NUM_ENVS="${NUM_ENVS}" \
    NUM_STEPS="${NUM_STEPS}" \
    UPDATE_EPOCHS="${UPDATE_EPOCHS}" \
    NUM_MINIBATCHES="${NUM_MINIBATCHES}" \
    TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
    LR="${LR}" \
    WANDB_MODE="${WANDB_MODE}" \
    ENTITY="${WANDB_ENTITY}" \
    PROJECT="${WANDB_PROJECT}" \
    +STDOUT_LOG=true +STDOUT_SUMMARY=true

echo
echo "=== done at $(date) ==="
