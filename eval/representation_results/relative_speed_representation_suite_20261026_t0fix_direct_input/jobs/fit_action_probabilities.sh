#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
export XLA_FLAGS=""
export MPLCONFIGDIR=/juice6/u/jshe/emergent_partner_grid/eval/representation_results/relative_speed_representation_suite_20261026_t0fix_direct_input/mpl_cache
exec /nlp/scr/jshe/miniconda3/envs/emergent_partner_model/bin/python -u /juice6/u/jshe/emergent_partner_grid/eval/representation_results/relative_speed_representation_suite_20261026_t0fix_direct_input/action_probabilities/source/representation_probe_suite.py worker /juice6/u/jshe/emergent_partner_grid/eval/representation_results/relative_speed_representation_suite_20261026_t0fix_direct_input/action_probabilities/analysis_run.json --index "${SLURM_ARRAY_TASK_ID}"
