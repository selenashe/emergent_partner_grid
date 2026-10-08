#!/usr/bin/env bash
set -euo pipefail
source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate emergent_partner_model
# Match the original batch's deterministic GPU kernel settings. Categorical
# action draws remain random and reproducible from each policy's learner seed.
export TF_CUDNN_USE_AUTOTUNE=0
export TF_CUDNN_DETERMINISTIC=1
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
export PYTHONDONTWRITEBYTECODE=1
unset JAX_PLATFORMS
# The queued wrapper and runner are frozen together before job submission.
# "$@" supplies the stage, manifest, and (for training) version and policy tag.
runner_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python -u "${runner_dir}/extend_counterbalanced_training.py" "$@"
