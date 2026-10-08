"""Regression tests for uncapped exact-balance corpus generation."""

import csv
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_prep"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import build_final_corpus as builder
from capability_selection import evaluate_layout, passes_exact_allocation_balance
from env_generator import GridEnv, compute_metrics


def test_exact_balance_requires_no_ties_and_both_goals():
    stats = evaluate_layout(5, 5, 4, 4, builder.TRAIN_CAPABILITY_PAIRS, 500, 0.01, 1.0)
    assert passes_exact_allocation_balance(stats)
    assert not passes_exact_allocation_balance(replace(stats, n_opt_red=11, n_opt_blue=11, n_opt_ties=2))
    assert not passes_exact_allocation_balance(replace(stats, n_opt_red=13, n_opt_blue=11))
    assert not passes_exact_allocation_balance(replace(stats, n_opt_red=0, n_opt_blue=0, n_opt_ties=24))


def _candidates():
    candidates = []
    for row in (5, 6):
        for col in range(5):
            env = GridEnv(np.zeros((7, 7), dtype=np.int8), (3, col), (row, 3),
                          (0, 0), (0, 6), len(candidates), 0.0)
            env.metadata = compute_metrics(env, switching_k=2)
            candidates.append(env)
    # Deduplication remains independent of scientific selection.
    return candidates + [candidates[0]]


@pytest.mark.parametrize("capped", [False, True])
def test_builder_saves_every_exact_survivor_and_resolves_symmetries(tmp_path, monkeypatch, capped):
    monkeypatch.setattr(builder, "generate_envs", lambda **kwargs: _candidates())
    out = tmp_path / "corpus"
    arguments = ["build_final_corpus.py", "--out_dir", str(out),
                 "--train_val_test_ratio", "1", "0", "0", "--min_delta_reward", "0",
                 "--horizon_n_sigma", "100", "--skip_render", "--skip_plots"]
    if capped:
        arguments += ["--n_final", "8", "--centroid_p_opt_red_target", "0.5"]
    monkeypatch.setattr(sys, "argv", arguments)
    builder.main()
    manifest = json.loads((out / "manifest.json").read_text())
    expected = 8 if capped else 10
    assert manifest["n_candidates_generated"] == 11
    assert manifest["n_unique_candidates"] == 10
    assert manifest["n_after_horizon_filter"] == 10
    assert manifest["n_final"] == manifest["n_train"] == expected
    assert manifest["n_val"] == manifest["n_test"] == 0
    assert "n_after_basic_validity" not in manifest
    assert "n_after_feasibility" not in manifest
    assert "min_oracle_success" not in manifest["thresholds"]
    assert manifest["d4_invariance_flagged"] == 0
    counts = list(manifest["symmetry_counts"].values())
    assert sum(counts) == expected
    assert max(counts) - min(counts) <= 1
    assert manifest["per_sym"] == (1 if capped else None)
    assert ("--n_final" in manifest["reproduce_command"]) == capped
    assert "stratify_seed" not in manifest["seeds"]
    assert "--stratify_seed" not in manifest["reproduce_command"]
    assert "--n_bins_per_feature" not in manifest["reproduce_command"]
    with np.load(out / "train_layouts.npz") as saved:
        assert saved["layouts"].shape == (expected, 7, 7)
    paths = list((out / "layouts/train").glob("*.json"))
    assert len(paths) == expected
    for path in paths:
        meta = json.loads(path.read_text())["metadata"]
        selection = meta["selection"]
        assert selection["n_opt_red"] == selection["n_opt_blue"] == 12
        assert selection["n_opt_ties"] == 0
        assert "selection_pass_feasibility" not in meta
    rows = list(csv.DictReader((out / "diagnostics/candidates.csv").open()))
    assert sum(row["selected_final"] == "True" for row in rows) == expected
    assert "pass_feasibility" not in rows[0]
    assert "feasibility" not in {stage["stage"] for stage in manifest["filter_stages"]}
    # A repeated run cannot overwrite a corpus or leave stale layouts.
    with pytest.raises(SystemExit):
        builder.main()


@pytest.mark.parametrize("arguments, message", [
    (["--n_final", "8"], "--n_final requires --centroid_p_opt_red_target"),
    (["--stratify_seed", "2026"], "unrecognized arguments: --stratify_seed"),
    (["--n_bins_per_feature", "3"], "unrecognized arguments: --n_bins_per_feature"),
])
def test_builder_rejects_removed_stratification_before_generation(monkeypatch, capsys,
                                                                 arguments, message):
    def unexpected_generation(**kwargs):
        pytest.fail("Invalid selection options must fail before generating layouts")

    monkeypatch.setattr(builder, "generate_envs", unexpected_generation)
    monkeypatch.setattr(sys, "argv", ["build_final_corpus.py", *arguments])
    with pytest.raises(SystemExit) as error:
        builder.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
