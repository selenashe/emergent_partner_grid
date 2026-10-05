"""Standalone evaluation must use the checkpoint's training layout corpus."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eval import evaluate_partner_modelling as evaluator


@pytest.mark.parametrize("override", [None, "explicit_layout_override"])
def test_standalone_eval_uses_saved_layouts_unless_overridden(tmp_path, monkeypatch, override):
    saved_layouts = str(tmp_path / "balanced_layouts")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "ALLOCATION_PROTOCOL": "online_v2", "ENV_KWARGS": {"layouts_dir": saved_layouts},
    }))
    arguments = ["evaluate_partner_modelling.py", "--config", str(config_path),
                 "--params", "unused.safetensors", "--out_prefix", str(tmp_path / "eval")]
    if override:
        arguments.extend(["--layouts_dir", override])
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(evaluator, "load_params", lambda _: {})
    seen_layouts = []

    def rollout(*args, **kwargs):
        seen_layouts.append(kwargs["layouts_dir"])
        return {}, None

    monkeypatch.setattr(evaluator, "rollout_condition", rollout)
    monkeypatch.setattr(evaluator, "_save_hdf5", lambda *args: None)
    monkeypatch.setattr(evaluator, "summarize", lambda *args, **kwargs: {
        "round_success_rate": 0.0, "mean_reward_per_round": 0.0,
        "fraction_partner_assigned_faster": 0.0, "assignment_switches_per_round": 0.0,
    })
    evaluator.main()
    assert seen_layouts == [override or saved_layouts] * 2
