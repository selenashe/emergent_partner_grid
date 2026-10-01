#!/usr/bin/env bash
#SBATCH --job-name=cg_eval_all
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/emergent_partner_grid/logs/cg_eval_all_%j.out
#SBATCH --error=/juice6/u/jshe/emergent_partner_grid/logs/cg_eval_all_%j.err
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

REPO=/juice6/u/jshe/emergent_partner_grid
cd "$REPO"
mkdir -p dev/eval_out

for cfg in dev/train_logs/*_config.json; do
    base=$(basename "$cfg" _config.json)
    params="dev/train_logs/${base}.safetensors"
    out_prefix="dev/eval_out/${base}"
    if [ ! -f "$params" ]; then
        echo "skip $base — no params"; continue
    fi
    if [ -f "${out_prefix}_summary.json" ]; then
        echo "skip $base — already evaluated"; continue
    fi
    echo "=== eval $base ==="
    # Save hidden states only for RNN checkpoints.
    save_hidden_flag=""
    if [[ "$base" == rnn_* ]]; then
        save_hidden_flag="--save_hidden"
    fi
    python -u analysis/evaluate_partner_modelling.py \
        --config "$cfg" \
        --params "$params" \
        --n_episodes_per_capability 20 \
        --seed 12345 \
        --out_prefix "$out_prefix" \
        $save_hidden_flag
done
echo "=== all done ==="
