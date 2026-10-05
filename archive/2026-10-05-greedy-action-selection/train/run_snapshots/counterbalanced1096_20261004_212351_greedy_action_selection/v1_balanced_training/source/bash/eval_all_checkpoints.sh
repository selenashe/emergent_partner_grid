#!/usr/bin/env bash
#SBATCH --job-name=cg_eval_all
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/emergent_partner_grid/eval/slurm_logs/cg_eval_all_%j.out
#SBATCH --error=/juice6/u/jshe/emergent_partner_grid/eval/slurm_logs/cg_eval_all_%j.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --exclude=sphinx9

set -euo pipefail
source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate emergent_partner_model

# cuDNN determinism to avoid autotune races
export TF_CUDNN_USE_AUTOTUNE=0
export TF_CUDNN_DETERMINISTIC=1
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/emergent_partner_grid}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/train/train_logs/online_v2}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${REPO_ROOT}/eval/eval_out/online_v2}"
mkdir -p "${EVAL_OUT_DIR}"
EVALUATOR_SCRIPT="eval/evaluate_partner_modelling.py"
if [[ ! -f "${EVALUATOR_SCRIPT}" ]]; then
    EVALUATOR_SCRIPT="analysis/evaluate_partner_modelling.py"  # original frozen snapshots
fi

shopt -s nullglob
configs=("${CHECKPOINT_DIR}"/*_config.json)
if [[ "${EXPECTED_CHECKPOINTS:-0}" -gt 0 && "${#configs[@]}" -ne "${EXPECTED_CHECKPOINTS}" ]]; then
    echo "Expected ${EXPECTED_CHECKPOINTS} configs, found ${#configs[@]} in ${CHECKPOINT_DIR}" >&2
    exit 1
fi
for cfg in "${configs[@]}"; do
    base=$(basename "$cfg" _config.json)
    params="${CHECKPOINT_DIR}/${base}.safetensors"
    out_prefix="${EVAL_OUT_DIR}/${base}"
    if [ ! -f "$params" ]; then
        if [[ "${EXPECTED_CHECKPOINTS:-0}" -gt 0 ]]; then
            echo "Missing checkpoint: ${params}" >&2; exit 1
        fi
        echo "skip $base — no params"; continue
    fi
    if [ -f "${out_prefix}_summary.json" ]; then
        echo "skip $base — already evaluated"; continue
    fi
    echo "=== eval $base ==="
    # Save hidden states only for RNN checkpoints.
    save_hidden_args=()
    if [[ "$base" == rnn_* ]]; then
        save_hidden_args=(--save_hidden)
    fi
    layouts_args=()
    if [[ -n "${LAYOUTS_DIR:-}" ]]; then
        layouts_args=(--layouts_dir "${LAYOUTS_DIR}")
    fi
    python -u "${EVALUATOR_SCRIPT}" \
        --config "$cfg" \
        --params "$params" \
        --n_episodes_per_capability 20 \
        --seed 12345 \
        --out_prefix "$out_prefix" \
        "${save_hidden_args[@]}" "${layouts_args[@]}"
done
echo "=== all done ==="
