#!/usr/bin/env bash
#SBATCH --job-name=cg_capability
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/emergent_partner_grid/train/slurm_logs/cg_cap_%j.out
#SBATCH --error=/juice6/u/jshe/emergent_partner_grid/train/slurm_logs/cg_cap_%j.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --exclude=sphinx9
# ---------------------------------------------------------------------------
# CoordinationGrid final experiment launcher.
#
# One trainer, four experimental conditions selected via env vars:
#
#   MODEL_TYPE     = rnn | mlp
#   PARTNER_REGIME = diverse | single
#   INFLUENCE      = true | false
#
# Presets (from the replication design):
#
#   CONDITION=rnn_diverse_influence     -> multi-partner RNN (main)
#   CONDITION=mlp_diverse_influence     -> memory control
#   CONDITION=rnn_single_influence      -> partner-diversity control
#   CONDITION=rnn_diverse_noinfluence   -> influence-pressure control
#
# The default corpus is data_prep/grids_capability_selected_balanced_1096
# (1096 layouts, used for both training and evaluation). Training and test
# capability populations are defined in
# jaxmarl.environments.coordination_grid.capability_populations.
#
# Prepare and validate a counterbalanced batch with submit_counterbalanced_training.py.
# Its manifest supplies REPO_ROOT pointing at the matching prepared source snapshot.
# ---------------------------------------------------------------------------
set -euo pipefail

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate emergent_partner_model

# Match the evaluator's cuDNN settings after earlier convolution failures.
export TF_CUDNN_USE_AUTOTUNE=0
export TF_CUDNN_DETERMINISTIC=1
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_deterministic_ops=true"

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/emergent_partner_grid}"
cd "${REPO_ROOT}" || exit 1

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/train/train_logs/counterbalanced}"
HYDRA_OUTPUT_DIR="${HYDRA_OUTPUT_DIR:-${REPO_ROOT}/train/hydra_outputs}"
mkdir -p "${CHECKPOINT_DIR}" "${HYDRA_OUTPUT_DIR}"

# Audit guide: translate CONDITION into the three scientific switches below.
# This case block chooses memory, partner diversity, and causal influence;
# SEED changes learner randomness. The partner itself remains scripted.
# --- Condition preset -------------------------------------------------------
CONDITION="${CONDITION:-rnn_diverse_influence}"
case "${CONDITION}" in
    rnn_diverse_influence)
        MODEL_TYPE="rnn";  PARTNER_REGIME="diverse"; INFLUENCE="true";;
    mlp_diverse_influence)
        MODEL_TYPE="mlp";  PARTNER_REGIME="diverse"; INFLUENCE="true";;
    rnn_single_influence)
        MODEL_TYPE="rnn";  PARTNER_REGIME="single";  INFLUENCE="true";;
    rnn_diverse_noinfluence)
        MODEL_TYPE="rnn";  PARTNER_REGIME="diverse"; INFLUENCE="false";;
    *)
        echo "unknown CONDITION=${CONDITION}"; exit 2;;
esac

# --- Overridable hyperparameters -------------------------------------------
SEED="${SEED:-1}"
NUM_SEEDS="${NUM_SEEDS:-1}"
NUM_ENVS="${NUM_ENVS:-256}"
NUM_STEPS="${NUM_STEPS:-256}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-64}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-60000000}"
LR="${LR:-5e-4}"
MAX_STEPS="${MAX_STEPS:-100}"
STEP_PENALTY="${STEP_PENALTY:-0.01}"
ROUNDS_PER_EPISODE="${ROUNDS_PER_EPISODE:-20}"
HIDE_PARTNER_UNTIL_TIME="${HIDE_PARTNER_UNTIL_TIME:-0}"
SCHEDULE_SEED="${SCHEDULE_SEED:-2026}"
N_EPS_TOTAL="${N_EPS_TOTAL:-131072}"
EVAL_EPISODES_PER_CAPABILITY="${EVAL_EPISODES_PER_CAPABILITY:-20}"

LAYOUTS_DIR="${LAYOUTS_DIR:-${REPO_ROOT}/data_prep/grids_capability_selected_balanced_1096/layouts/train}"
N_LAYOUTS="$(ls -1 "${LAYOUTS_DIR}"/*.json 2>/dev/null | wc -l)"

TAG="${TAG:-${CONDITION}_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
SAVE_PARAMS_PATH="${CHECKPOINT_DIR}/${TAG}.safetensors"

WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-coordination_grid}"

echo "===================================================================="
echo "  CoordinationGrid run  (job=${SLURM_JOB_ID:-local})"
echo "===================================================================="
echo "  CONDITION               : ${CONDITION}"
echo "  MODEL_TYPE              : ${MODEL_TYPE}"
echo "  PARTNER_REGIME          : ${PARTNER_REGIME}"
echo "  INFLUENCE               : ${INFLUENCE}"
echo "  rounds_per_episode      : ${ROUNDS_PER_EPISODE}"
echo "  max_steps               : ${MAX_STEPS}"
echo "  step_penalty            : ${STEP_PENALTY}"
echo "  layouts_dir             : ${LAYOUTS_DIR}  (n=${N_LAYOUTS})"
echo "  SEED / NUM_SEEDS        : ${SEED} / ${NUM_SEEDS}"
echo "  N_EPS_TOTAL             : ${N_EPS_TOTAL}"
echo "  TOTAL_TIMESTEPS         : ${TOTAL_TIMESTEPS}"
echo "  NUM_ENVS / NUM_STEPS    : ${NUM_ENVS} / ${NUM_STEPS}"
echo "  NUM_MINIBATCHES         : ${NUM_MINIBATCHES}"
echo "  SAVE_PARAMS             : ${SAVE_PARAMS_PATH}"
echo "  GPU visible:"
nvidia-smi -L 2>/dev/null || echo "    (no nvidia-smi)"
echo

# Audit guide: execute the chosen trainer using explicit Hydra overrides.
# REPO_ROOT can select an immutable source snapshot; the fallback supports
# historical paths inside snapshots. The companion saved JSON configuration,
# rather than these present defaults, identifies an already trained policy.
TRAIN_SCRIPT="train/ippo_rnn_coordination_grid.py"
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    TRAIN_SCRIPT="baselines/IPPO/ippo_rnn_coordination_grid.py"  # original frozen snapshots
fi
python -u "${TRAIN_SCRIPT}" \
    SEED="${SEED}" \
    NUM_SEEDS="${NUM_SEEDS}" \
    MODEL_TYPE="${MODEL_TYPE}" \
    PARTNER_REGIME="${PARTNER_REGIME}" \
    INFLUENCE="${INFLUENCE}" \
    ALLOCATION_PROTOCOL="${ALLOCATION_PROTOCOL:-online_v2}" \
    ENV_KWARGS.layouts_dir="${LAYOUTS_DIR}" \
    ENV_KWARGS.rounds_per_episode="${ROUNDS_PER_EPISODE}" \
    ENV_KWARGS.max_steps="${MAX_STEPS}" \
    ENV_KWARGS.step_penalty="${STEP_PENALTY}" \
    ENV_KWARGS.hide_partner_until_time="${HIDE_PARTNER_UNTIL_TIME}" \
    ENV_KWARGS.communication_condition="action_only" \
    EVAL_EPISODES_PER_CAPABILITY="${EVAL_EPISODES_PER_CAPABILITY}" \
    SAVE_PARAMS_PATH="${SAVE_PARAMS_PATH}" \
    NUM_ENVS="${NUM_ENVS}" \
    NUM_STEPS="${NUM_STEPS}" \
    UPDATE_EPOCHS="${UPDATE_EPOCHS}" \
    NUM_MINIBATCHES="${NUM_MINIBATCHES}" \
    TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
    LR="${LR}" \
    N_EPS_TOTAL="${N_EPS_TOTAL}" \
    SCHEDULE_SEED="${SCHEDULE_SEED}" \
    WANDB_MODE="${WANDB_MODE}" \
    ENTITY="${WANDB_ENTITY}" \
    PROJECT="${WANDB_PROJECT}" \
    hydra.run.dir="${HYDRA_OUTPUT_DIR}/${TAG}" \
    +STDOUT_LOG=true +STDOUT_SUMMARY=true

echo
echo "=== done at $(date) ==="
