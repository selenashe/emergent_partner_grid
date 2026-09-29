#!/usr/bin/env bash
#SBATCH --job-name=cg_capability
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/emergent_partner_grid/logs/cg_cap_%j.out
#SBATCH --error=/juice6/u/jshe/emergent_partner_grid/logs/cg_cap_%j.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
# ---------------------------------------------------------------------------
# Capability-vector CoordinationGrid experiment.
#
# The environment is now the (c_R, c_B) capability build (see
# jaxmarl.environments.coordination_grid). The 30 training capability pairs
# and 6 held-out pairs live as constants in that module; the config yaml
# pins the training list explicitly. This launcher just runs the trainer.
#
# Usage:
#   sbatch bash/train_final_experiment.sh
# ---------------------------------------------------------------------------
set -euo pipefail

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate emergent_partner_model

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/emergent_partner_grid}"
cd "${REPO_ROOT}" || exit 1

mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/dev/train_logs"

# --- Overridable hyperparameters ---
SEED="${SEED:-1}"
NUM_SEEDS="${NUM_SEEDS:-1}"
NUM_ENVS="${NUM_ENVS:-256}"
NUM_STEPS="${NUM_STEPS:-256}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-8}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-60000000}"
LR="${LR:-5e-4}"
MAX_STEPS="${MAX_STEPS:-64}"
STEP_PENALTY="${STEP_PENALTY:-0.02}"
ROUNDS_PER_EPISODE="${ROUNDS_PER_EPISODE:-20}"
AUGMENT_SYMMETRIES="${AUGMENT_SYMMETRIES:-false}"
HIDE_PARTNER_UNTIL_TIME="${HIDE_PARTNER_UNTIL_TIME:-0}"
N_SWEEPS="${N_SWEEPS:-256}"
SCHEDULE_SEED="${SCHEDULE_SEED:-2026}"
EVAL_EPISODES_PER_CAPABILITY="${EVAL_EPISODES_PER_CAPABILITY:-16}"

TRAIN_LAYOUTS="${TRAIN_LAYOUTS:-${REPO_ROOT}/dev/grids_final/layouts/train}"
VAL_LAYOUTS="${VAL_LAYOUTS:-${REPO_ROOT}/dev/grids_final/layouts/val}"
TEST_LAYOUTS="${TEST_LAYOUTS:-${REPO_ROOT}/dev/grids_final/layouts/test}"

N_TRAIN="$(ls -1 "${TRAIN_LAYOUTS}"/*.json 2>/dev/null | wc -l)"
N_VAL="$(ls -1 "${VAL_LAYOUTS}"/*.json 2>/dev/null | wc -l)"
N_TEST="$(ls -1 "${TEST_LAYOUTS}"/*.json 2>/dev/null | wc -l)"

TAG="${TAG:-capability_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
SAVE_PARAMS_PATH="${REPO_ROOT}/dev/train_logs/${TAG}.safetensors"

WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-coordination_grid}"

echo "===================================================================="
echo "  Capability experiment  (job=${SLURM_JOB_ID:-local})"
echo "===================================================================="
echo "  rounds_per_episode      : ${ROUNDS_PER_EPISODE}"
echo "  hide_partner_until_time : ${HIDE_PARTNER_UNTIL_TIME}"
echo "  augment_symmetries      : ${AUGMENT_SYMMETRIES}"
echo "  train / val / test      : ${N_TRAIN} / ${N_VAL} / ${N_TEST}"
echo "  n_sweeps                : ${N_SWEEPS}"
echo "  schedule_seed           : ${SCHEDULE_SEED}"
echo "  TOTAL_TIMESTEPS         : ${TOTAL_TIMESTEPS}"
echo "  NUM_ENVS / NUM_STEPS    : ${NUM_ENVS} / ${NUM_STEPS}"
echo "  SAVE_PARAMS             : ${SAVE_PARAMS_PATH}"
echo "  GPU visible:"
nvidia-smi -L 2>/dev/null || echo "    (no nvidia-smi)"
echo

python -u baselines/IPPO/ippo_rnn_coordination_grid.py \
    SEED="${SEED}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    ENV_KWARGS.layouts_dir="${TRAIN_LAYOUTS}" \
    ENV_KWARGS.rounds_per_episode="${ROUNDS_PER_EPISODE}" \
    ENV_KWARGS.max_steps="${MAX_STEPS}" \
    ENV_KWARGS.step_penalty="${STEP_PENALTY}" \
    ENV_KWARGS.augment_symmetries="${AUGMENT_SYMMETRIES}" \
    ENV_KWARGS.hide_partner_until_time="${HIDE_PARTNER_UNTIL_TIME}" \
    ENV_KWARGS.communication_condition="action_only" \
    EVAL_LAYOUTS_DIRS.val="${VAL_LAYOUTS}" \
    EVAL_LAYOUTS_DIRS.test="${TEST_LAYOUTS}" \
    EVAL_EPISODES_PER_CAPABILITY="${EVAL_EPISODES_PER_CAPABILITY}" \
    SAVE_PARAMS_PATH="${SAVE_PARAMS_PATH}" \
    NUM_ENVS="${NUM_ENVS}" \
    NUM_STEPS="${NUM_STEPS}" \
    UPDATE_EPOCHS="${UPDATE_EPOCHS}" \
    NUM_MINIBATCHES="${NUM_MINIBATCHES}" \
    TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
    LR="${LR}" \
    "+N_SWEEPS=${N_SWEEPS}" \
    "+SCHEDULE_SEED=${SCHEDULE_SEED}" \
    WANDB_MODE="${WANDB_MODE}" \
    ENTITY="${WANDB_ENTITY}" \
    PROJECT="${WANDB_PROJECT}" \
    +STDOUT_LOG=true +STDOUT_SUMMARY=true

echo
echo "=== done at $(date) ==="
