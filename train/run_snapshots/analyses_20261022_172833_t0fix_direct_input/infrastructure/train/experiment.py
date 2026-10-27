"""Maintained five-seed direct-input cohort identities and saved inputs."""
from pathlib import Path
import os

ROOT = Path(os.environ.get('GRID_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
BATCH = "v1_matched_v2_v3_20261022_080155_t0fix_direct_input"
SEEDS = tuple(range(1, 6))
VERSIONS = ("v1_matched", "v2", "v3")
CONDITIONS = ("rnn_diverse_influence", "mlp_diverse_influence",
              "rnn_single_influence", "rnn_diverse_noinfluence")
TRAINING_MANIFEST = ROOT / "train/manifests" / ("sbatch_" + BATCH + ".json")
ANALYSIS_BATCH = "analyses_20261022_172833_t0fix_direct_input"
ANALYSIS_SOURCE = ROOT / "train/run_snapshots" / ANALYSIS_BATCH / "analysis_source_t0fix_direct_input"
VIEW_MANIFEST = ANALYSIS_SOURCE / "train/manifests" / ("sbatch_" + BATCH + ".json")
INPUT_MANIFEST = ROOT / "train/run_snapshots" / ANALYSIS_BATCH / "representation_inputs_t0fix_direct_input.json"
