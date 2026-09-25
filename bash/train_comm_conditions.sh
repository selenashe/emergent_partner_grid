#!/usr/bin/env bash
#SBATCH --job-name=cg_comm_cond
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/emergent_partner_grid/logs/cg_comm_%j.out
#SBATCH --error=/juice6/u/jshe/emergent_partner_grid/logs/cg_comm_%j.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
# ---------------------------------------------------------------------------
# 3-condition communication experiment on the Stage C+D pipeline.
# Run one condition per invocation, selected by ``COMM_CONDITION``:
#
#   action_only      : no message channel; partner samples 50/50 regardless
#                      of z; policy mask allows only STAY+NONE at t=0.
#   universal        : messages have globally fixed meanings (M0→RED w.p.1,
#                      M1→BLUE w.p.1); decoder ignores z. Policy mask allows
#                      STAY+{NONE,M0,M1} at t=0.
#   partner_specific : messages are z-conditional (P(RED|M0,z)=z,
#                      P(RED|M1,z)=1-z, P(RED|NONE)=0.5). Policy mask same
#                      as ``universal``.
#
# All other hyperparameters — layouts pool, D4 aug, z pool, rounds/episode,
# max_steps, hide_partner_until_time, PPO settings, network — are matched
# across conditions. The ONLY substantive experimental difference is the
# semantics/availability of the t=0 communication channel.
#
# Usage:
#   sbatch --export=ALL,COMM_CONDITION=action_only      bash/train_comm_conditions.sh
#   sbatch --export=ALL,COMM_CONDITION=universal        bash/train_comm_conditions.sh
#   sbatch --export=ALL,COMM_CONDITION=partner_specific bash/train_comm_conditions.sh
# ---------------------------------------------------------------------------
set -euo pipefail

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate emergent_partner_model

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/emergent_partner_grid}"
cd "${REPO_ROOT}" || exit 1

mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/dev/train_logs"

# --- Required: which condition to run ---
COMM_CONDITION="${COMM_CONDITION:?Set COMM_CONDITION to one of action_only|universal|partner_specific}"
case "${COMM_CONDITION}" in
    action_only|universal|partner_specific) ;;
    *) echo "ERROR: COMM_CONDITION must be one of action_only|universal|partner_specific (got '${COMM_CONDITION}')" >&2; exit 2;;
esac

# --- Hyperparameters (env-overridable; MATCHED across conditions) ---
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
PARTNER_Z_VALUES="${PARTNER_Z_VALUES:-[0.2,0.4,0.6,0.8]}"
AUGMENT_SYMMETRIES="${AUGMENT_SYMMETRIES:-true}"
HIDE_PARTNER_UNTIL_TIME="${HIDE_PARTNER_UNTIL_TIME:-0}"
EVAL_EPISODES_PER_Z="${EVAL_EPISODES_PER_Z:-64}"

TRAIN_LAYOUTS="${TRAIN_LAYOUTS:-${REPO_ROOT}/dev/grids/layouts/train}"
VAL_LAYOUTS="${VAL_LAYOUTS:-${REPO_ROOT}/dev/grids/layouts/val}"
TEST_LAYOUTS="${TEST_LAYOUTS:-${REPO_ROOT}/dev/grids/layouts/test}"

# Condition-tagged run name so params/logs never collide.
TAG="${TAG:-comm_${COMM_CONDITION}_R${ROUNDS_PER_EPISODE}_hideK${HIDE_PARTNER_UNTIL_TIME}_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
SAVE_PARAMS_PATH="${REPO_ROOT}/dev/train_logs/${TAG}.safetensors"

WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-coordination_grid}"

echo "=== 3-condition comm experiment  (job=${SLURM_JOB_ID:-local}, tag=${TAG}) ==="
echo "COMM_CONDITION      = ${COMM_CONDITION}"
echo "REPO_ROOT           = ${REPO_ROOT}"
echo "TRAIN_LAYOUTS       = ${TRAIN_LAYOUTS}"
echo "VAL_LAYOUTS         = ${VAL_LAYOUTS}"
echo "TEST_LAYOUTS        = ${TEST_LAYOUTS}"
echo "AUGMENT_SYM         = ${AUGMENT_SYMMETRIES}"
echo "HIDE_PARTNER_UNTIL  = ${HIDE_PARTNER_UNTIL_TIME}"
echo "PARTNER_Z_VALUES    = ${PARTNER_Z_VALUES}"
echo "ROUNDS_PER_EPISODE  = ${ROUNDS_PER_EPISODE}"
echo "MAX_STEPS/round     = ${MAX_STEPS}"
echo "TOTAL_TIMESTEPS     = ${TOTAL_TIMESTEPS}"
echo "NUM_ENVS            = ${NUM_ENVS}   NUM_STEPS=${NUM_STEPS}"
echo "SAVE_PARAMS         = ${SAVE_PARAMS_PATH}"
echo "WANDB_MODE          = ${WANDB_MODE}"
echo "GPU visible:"
nvidia-smi -L 2>/dev/null || echo "  (no nvidia-smi)"
echo

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
    WANDB_MODE="${WANDB_MODE}" \
    ENTITY="${WANDB_ENTITY}" \
    PROJECT="${WANDB_PROJECT}" \
    +STDOUT_LOG=true +STDOUT_SUMMARY=true

echo
echo "=== done at $(date) ==="
