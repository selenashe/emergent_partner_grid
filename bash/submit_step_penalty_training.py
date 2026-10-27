"""Compatibility exports for matching frozen-source checkpoint evaluation.

The maintained launcher uses train.experiment_runtime. Original-CNN-derived
step-penalty preparation/submission functions are retired; saved scientific
sources for completed experiments remain in their frozen snapshots.
"""
from train.experiment_runtime import publish, evaluation_command, evaluate_job
