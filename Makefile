PYTHON ?= python

.PHONY: help test prepare-direct-input

help:
	@echo 'test: run active CoordinationGrid regression tests'
	@echo 'prepare-direct-input: prepare the five-seed t0-fixed direct-input experiments (no submission)'

test:
	JAX_PLATFORMS=cpu $(PYTHON) -m pytest tests/coordination_grid/test_counterbalanced_scheduler.py tests/coordination_grid/test_evaluation_layouts.py tests/coordination_grid/test_online_allocation.py tests/data_prep/test_capability_selection.py tests/analysis/test_representation_analysis.py

prepare-direct-input:
	$(PYTHON) bash/submit_t0fix_direct_input_training.py prepare
