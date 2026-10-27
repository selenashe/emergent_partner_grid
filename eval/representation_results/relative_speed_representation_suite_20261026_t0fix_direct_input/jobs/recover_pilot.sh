#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset JAX_PLATFORMS
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
exec /nlp/scr/jshe/miniconda3/envs/emergent_partner_model/bin/python -u /juice6/u/jshe/emergent_partner_grid/eval/representation_probe_suite.py recover /juice6/u/jshe/emergent_partner_grid/eval/representation_results/relative_speed_representation_suite_20261026_t0fix_direct_input/suite_run.json --index 0 --only additional --diagnostic
