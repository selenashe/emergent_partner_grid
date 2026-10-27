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
# Slurm copies this shell script into its spool directory. Its BASH_SOURCE
# therefore cannot locate the shared Python file. Submission supplies the
# frozen runner's absolute path as the first argument for every stage.
runner_path="${1:?Expected the absolute frozen Python runner path}"
shift
if [[ "${runner_path}" != /* || ! -f "${runner_path}" ]]; then
    echo "Missing absolute runner file: ${runner_path}" >&2
    exit 2
fi
python -u "${runner_path}" "$@"
