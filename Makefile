PYTHON ?= python

.PHONY: help test prepare-counterbalanced

help:
	@echo 'test: run active CoordinationGrid regression tests'
	@echo 'prepare-counterbalanced: freeze v1/v2 sources and schedules (no Slurm submission)'

test:
	JAX_PLATFORMS=cpu $(PYTHON) -m pytest tests/coordination_grid/test_counterbalanced_scheduler.py tests/coordination_grid/test_evaluation_layouts.py tests/coordination_grid/test_online_allocation.py tests/data_prep/test_capability_selection.py tests/analysis/test_representation_analysis.py

prepare-counterbalanced:
	$(PYTHON) bash/submit_counterbalanced_training.py --prepare
