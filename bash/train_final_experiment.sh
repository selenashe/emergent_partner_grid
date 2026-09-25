#!/usr/bin/env bash
#SBATCH --job-name=cg_final
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/emergent_partner_grid/logs/cg_final_%A_%a.out
#SBATCH --error=/juice6/u/jshe/emergent_partner_grid/logs/cg_final_%A_%a.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --array=0-2
# ---------------------------------------------------------------------------
# Final experiment: 3-condition communication study over the balanced
# (z × layout) sweep schedule.
#
# Job-array indices:
#   0 -> action_only
#   1 -> universal
#   2 -> partner_specific
#
# Shared, matched hyperparameters (see also
# baselines/IPPO/config/ippo_rnn_coordination_grid.yaml):
#   * partner_z_values     : [0.1, 0.3, 0.5, 0.7, 0.9]
#   * rounds_per_episode   : 20
#   * hide_partner_until_time : 0
#   * augment_symmetries   : false (final corpus already has one D4/layout)
#   * layouts_dir (train/val/test) : dev/grids_final/layouts/{train,val,test}
#   * NUM_ENVS = 256, NUM_STEPS = 128, TOTAL_TIMESTEPS = 60_000_000
#
# Sweep schedule (built once per run inside make_train, seed = SCHEDULE_SEED):
#   n_z = 5, n_layouts_train = 1600, R = 20
#   pairings_per_sweep = 8000
#   episodes_per_sweep = 400
#   episodes_per_z_per_sweep = 80
# The trainer prints the sanity-check summary at startup.
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

# --- Resolve which condition this array-task runs ---
CONDITIONS=(action_only universal partner_specific)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX < 0 || IDX >= ${#CONDITIONS[@]} )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${IDX} out of range 0..$((${#CONDITIONS[@]}-1))" >&2
    exit 2
fi
COMM_CONDITION="${CONDITIONS[$IDX]}"

# --- Matched hyperparameters (env-overridable) ---
SEED="${SEED:-1}"
NUM_SEEDS="${NUM_SEEDS:-1}"
NUM_ENVS="${NUM_ENVS:-256}"
NUM_STEPS="${NUM_STEPS:-128}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-8}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-60000000}"
LR="${LR:-5e-4}"
MAX_STEPS="${MAX_STEPS:-15}"
ROUNDS_PER_EPISODE="${ROUNDS_PER_EPISODE:-20}"
PARTNER_Z_VALUES="${PARTNER_Z_VALUES:-[0.1,0.3,0.5,0.7,0.9]}"
AUGMENT_SYMMETRIES="${AUGMENT_SYMMETRIES:-false}"
HIDE_PARTNER_UNTIL_TIME="${HIDE_PARTNER_UNTIL_TIME:-0}"
N_SWEEPS="${N_SWEEPS:-256}"                  # matches NUM_ENVS so each worker starts on a distinct sweep
SCHEDULE_SEED="${SCHEDULE_SEED:-2026}"
EVAL_EPISODES_PER_Z="${EVAL_EPISODES_PER_Z:-64}"

TRAIN_LAYOUTS="${TRAIN_LAYOUTS:-${REPO_ROOT}/dev/grids_final/layouts/train}"
VAL_LAYOUTS="${VAL_LAYOUTS:-${REPO_ROOT}/dev/grids_final/layouts/val}"
TEST_LAYOUTS="${TEST_LAYOUTS:-${REPO_ROOT}/dev/grids_final/layouts/test}"

N_TRAIN="$(ls -1 "${TRAIN_LAYOUTS}"/*.json 2>/dev/null | wc -l)"
N_VAL="$(ls -1 "${VAL_LAYOUTS}"/*.json 2>/dev/null | wc -l)"
N_TEST="$(ls -1 "${TEST_LAYOUTS}"/*.json 2>/dev/null | wc -l)"
Z_POOL_LEN=$(python -c "import ast; print(len(ast.literal_eval('${PARTNER_Z_VALUES}')))")
PAIRINGS_PER_SWEEP=$(( Z_POOL_LEN * N_TRAIN ))

TAG="${TAG:-final_${COMM_CONDITION}_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
SAVE_PARAMS_PATH="${REPO_ROOT}/dev/train_logs/${TAG}.safetensors"

WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-coordination_grid}"

echo "===================================================================="
echo "  Final experiment  (job=${SLURM_JOB_ID:-local}/${SLURM_ARRAY_TASK_ID:-.})"
echo "===================================================================="
echo "  communication condition : ${COMM_CONDITION}"
echo "  partner_z_values        : ${PARTNER_Z_VALUES}     (|z|=${Z_POOL_LEN})"
echo "  rounds_per_episode      : ${ROUNDS_PER_EPISODE}"
echo "  hide_partner_until_time : ${HIDE_PARTNER_UNTIL_TIME}"
echo "  augment_symmetries      : ${AUGMENT_SYMMETRIES}"
echo "  train / val / test      : ${N_TRAIN} / ${N_VAL} / ${N_TEST}"
echo "  expected pairings/sweep : ${PAIRINGS_PER_SWEEP}  (must be 8000)"
echo "  n_sweeps                : ${N_SWEEPS}"
echo "  schedule_seed           : ${SCHEDULE_SEED}"
echo "  layouts (train)         : ${TRAIN_LAYOUTS}"
echo "  layouts (val)           : ${VAL_LAYOUTS}"
echo "  layouts (test)          : ${TEST_LAYOUTS}"
echo "  TOTAL_TIMESTEPS         : ${TOTAL_TIMESTEPS}"
echo "  NUM_ENVS / NUM_STEPS    : ${NUM_ENVS} / ${NUM_STEPS}"
echo "  SAVE_PARAMS             : ${SAVE_PARAMS_PATH}"
echo "  GPU visible:"
nvidia-smi -L 2>/dev/null || echo "    (no nvidia-smi)"
echo

# Guard: matched pairings/sweep check.
if [ "${PAIRINGS_PER_SWEEP}" -ne 8000 ]; then
    echo "WARNING: pairings_per_sweep=${PAIRINGS_PER_SWEEP} != 8000 " \
         "— check |z| × N_TRAIN." >&2
fi

python -u baselines/IPPO/ippo_rnn_coordination_grid.py \
    SEED="${SEED}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    ENV_KWARGS.layouts_dir="${TRAIN_LAYOUTS}" \
    ENV_KWARGS.partner_z_values="${PARTNER_Z_VALUES}" \
    ENV_KWARGS.rounds_per_episode="${ROUNDS_PER_EPISODE}" \
    ENV_KWARGS.max_steps="${MAX_STEPS}" \
    ENV_KWARGS.augment_symmetries="${AUGMENT_SYMMETRIES}" \
    ENV_KWARGS.hide_partner_until_time="${HIDE_PARTNER_UNTIL_TIME}" \
    ENV_KWARGS.communication_condition="${COMM_CONDITION}" \
    EVAL_LAYOUTS_DIRS.val="${VAL_LAYOUTS}" \
    EVAL_LAYOUTS_DIRS.test="${TEST_LAYOUTS}" \
    EVAL_EPISODES_PER_Z="${EVAL_EPISODES_PER_Z}" \
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
