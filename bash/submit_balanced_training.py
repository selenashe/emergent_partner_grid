"""Prepare a frozen online_v2 batch, then submit training and dependent eval.

    python bash/submit_balanced_training.py --prepare
    python bash/submit_balanced_training.py --submit-manifest train/manifests/<manifest>.json

The default batch is four conditions x seeds 1–5, using the 1096-layout
balanced corpus. Every successful submission is immediately recorded so
an interrupted launch can resume without duplicating jobs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repo_paths import resolve_record_paths
CONDITIONS = ("rnn_diverse_influence", "mlp_diverse_influence",
              "rnn_single_influence", "rnn_diverse_noinfluence")


def _write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def prepare():
    now = datetime.now(timezone.utc)
    batch = "balanced1096_" + now.strftime("%Y%m%d_%H%M%S")
    source_corpus = ROOT / "data_prep/grids_capability_selected_balanced_1096"
    layouts = sorted((source_corpus / "layouts/train").glob("*.json"))
    if len(layouts) != 1096:
        raise ValueError(f"Expected 1096 layouts, found {len(layouts)}")
    red = Counter()
    blue = Counter()
    for path in layouts:
        metadata = json.loads(path.read_text())["metadata"]
        red[metadata["ego_to_red"]] += 1
        blue[metadata["ego_to_blue"]] += 1
    if red != blue or red != Counter({i: 137 for i in range(1, 9)}):
        raise ValueError("Layout corpus does not have the requested distance marginals")

    snapshot = ROOT / "train/run_snapshots" / batch / "source"
    snapshot.mkdir(parents=True, exist_ok=False)
    source_files = []
    for directory in ("jaxmarl", "train/config", "eval", "tests/coordination_grid"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".yaml", ".yml"):
                source_files.append(path)
    source_files.extend(sorted((ROOT / "train").glob("*.py")))
    source_files.append(ROOT / "repo_paths.py")
    source_files.extend(ROOT / relative for relative in (
        "data_prep/capability_selection.py", "bash/train_final_experiment.sh",
        "bash/eval_all_checkpoints.sh", "bash/submit_balanced_training.py"))
    hashes = {}
    for path in source_files:
        relative = path.relative_to(ROOT)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    snapshot_corpus = snapshot / source_corpus.relative_to(ROOT)
    shutil.copytree(source_corpus / "layouts", snapshot_corpus / "layouts")
    for name in ("manifest.json", "train_layouts.npz"):
        shutil.copyfile(source_corpus / name, snapshot_corpus / name)
    # Hash layouts as one deterministic aggregate in sorted filename order.
    digest = hashlib.sha256()
    for path in layouts:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    checkpoint_dir = ROOT / "train/train_logs/online_v2" / batch
    eval_dir = ROOT / "eval/eval_out/online_v2" / batch
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    eval_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = ROOT / "train/manifests" / f"sbatch_online_v2_{batch}.json"
    common = {
        "REPO_ROOT": str(snapshot), "CHECKPOINT_DIR": str(checkpoint_dir),
        "HYDRA_OUTPUT_DIR": str(ROOT / "train/hydra_outputs" / batch),
        "LAYOUTS_DIR": str(snapshot_corpus / "layouts/train"),
        "NUM_SEEDS": "1", "NUM_ENVS": "256", "NUM_STEPS": "256",
        "UPDATE_EPOCHS": "4", "NUM_MINIBATCHES": "64", "TOTAL_TIMESTEPS": "60000000",
        "LR": "5e-4", "MAX_STEPS": "100", "STEP_PENALTY": "0.01",
        "ROUNDS_PER_EPISODE": "20", "HIDE_PARTNER_UNTIL_TIME": "0",
        "SCHEDULE_SEED": "2026", "N_EPS_TOTAL": "131072",
        "EVAL_EPISODES_PER_CAPABILITY": "20", "WANDB_MODE": "disabled",
        "XLA_FLAGS": "", "JAX_PLATFORMS": "", "PYTHONDONTWRITEBYTECODE": "1",
    }
    jobs = []
    for condition in CONDITIONS:
        for seed in range(1, 6):
            tag = f"{condition}_seed{seed}"
            export = common | {"CONDITION": condition, "SEED": str(seed), "TAG": tag}
            jobs.append({"condition": condition, "seed": seed, "tag": tag,
                         "command": ["sbatch", "--parsable", f"--chdir={snapshot}",
                                     f"--job-name=cg_b1096_{tag}",
                                     "--export=ALL," + ",".join(f"{k}={v}" for k, v in export.items()),
                                     str(snapshot / "bash/train_final_experiment.sh")]})
    manifest = {
        "prepared_at_utc": now.isoformat(), "status": "prepared", "batch": batch,
        "allocation_protocol": "online_v2", "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_corpus": str(source_corpus), "frozen_source_root": str(snapshot),
        "frozen_layouts_dir": common["LAYOUTS_DIR"], "source_sha256": hashes,
        "layout_files_sha256": digest.hexdigest(), "n_layouts": 1096,
        "red_distance_counts_1_to_8": [137] * 8, "blue_distance_counts_1_to_8": [137] * 8,
        "checkpoint_root": str(checkpoint_dir), "evaluation_root": str(eval_dir),
        "training": {"conditions": list(CONDITIONS), "seeds": list(range(1, 6)),
                     "nominal_timesteps_per_policy": 60000000,
                     "effective_timesteps_per_policy": 59965440, "environment": common},
        "resources_per_training_job": {"account": "nlp", "partition": "sphinx",
                                       "gpus": 1, "constraint": "80G", "memory": "32G",
                                       "cpus": 4, "time_limit": "08:00:00", "exclude": "sphinx9"},
        "evaluation": {"episodes_per_capability": 20, "seed": 12345,
                       "familiar_profiles": 24, "novel_profiles": 22,
                       "save_hidden_for_rnn": True, "time_limit": "02:00:00"},
        "training_jobs": jobs, "evaluation_jobs": [],
    }
    _write(manifest_path, manifest)
    print(manifest_path, flush=True)


def submit(path):
    manifest = json.loads(path.read_text())
    manifest = resolve_record_paths(manifest)
    snapshot = Path(manifest["frozen_source_root"])
    for name, expected in manifest["source_sha256"].items():
        if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source changed: {name}")
    digest = hashlib.sha256()
    for layout in sorted(Path(manifest["frozen_layouts_dir"]).glob("*.json")):
        digest.update(layout.name.encode()); digest.update(layout.read_bytes())
    if digest.hexdigest() != manifest["layout_files_sha256"]:
        raise ValueError("Frozen layout corpus changed")
    manifest["status"] = "submitting"
    _write(path, manifest)
    environment = os.environ.copy()
    # Avoid inheriting the interactive allocation or our CPU-only test setting.
    for key in list(environment):
        if key.startswith("SLURM_") or key in ("JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES"):
            environment.pop(key)
    for job in manifest["training_jobs"]:
        if "job_id" in job:
            continue
        output = subprocess.check_output(job["command"], cwd=ROOT, env=environment, text=True).strip()
        job["job_id"] = output.split(";")[0]
        if not job["job_id"].isdigit():
            raise RuntimeError(f"Unexpected sbatch response: {output}")
        _write(path, manifest)
        print(f"training {job['condition']} seed {job['seed']}: {job['job_id']}", flush=True)
    if not manifest["evaluation_jobs"]:
        dependency = "afterok:" + ":".join(job["job_id"] for job in manifest["training_jobs"])
        exports = {"REPO_ROOT": str(snapshot), "CHECKPOINT_DIR": manifest["checkpoint_root"],
                   "EVAL_OUT_DIR": manifest["evaluation_root"], "EXPECTED_CHECKPOINTS": "20",
                   "XLA_FLAGS": "", "JAX_PLATFORMS": "", "PYTHONDONTWRITEBYTECODE": "1"}
        command = ["sbatch", "--parsable", f"--chdir={snapshot}", "--job-name=cg_b1096_eval_all",
                   f"--dependency={dependency}", "--kill-on-invalid-dep=yes",
                   "--export=ALL," + ",".join(f"{k}={v}" for k, v in exports.items()),
                   str(snapshot / "bash/eval_all_checkpoints.sh")]
        output = subprocess.check_output(command, cwd=ROOT, env=environment, text=True).strip()
        job_id = output.split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError(f"Unexpected sbatch response: {output}")
        manifest["evaluation_jobs"].append({"job_id": job_id, "dependency": dependency, "command": command})
        _write(path, manifest)
        print(f"evaluation: {job_id} ({dependency})", flush=True)
    manifest["status"] = "submitted"
    manifest["submitted_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write(path, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--submit-manifest", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        submit(args.submit_manifest)
