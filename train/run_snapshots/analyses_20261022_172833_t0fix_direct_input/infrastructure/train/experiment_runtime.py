"""IO and frozen-source execution shared by the maintained experiment launchers."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

from repo_paths import artifact_path, artifact_root, resolve_path

PYTHON = "/nlp/scr/jshe/miniconda3/envs/emergent_partner_model/bin/python"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def environment(source, cpu=False):
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1",
               HYDRA_FULL_ERROR="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    directory = artifact_root() / "wandb"
    directory.mkdir(parents=True, exist_ok=True)
    env.setdefault("WANDB_DIR", str(directory))
    if cpu:
        env.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", XLA_FLAGS="")
    return env


def publish(directory, prefix):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for physical in artifact_path(directory).glob(prefix + "*"):
        logical = directory / physical.name
        if logical == physical:
            continue
        if logical.exists() and not logical.is_symlink():
            raise FileExistsError(logical)
        temporary = logical.with_name(logical.name + ".publishing")
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(physical)
        temporary.replace(logical)


def evaluation_command(manifest, job, smoke_target=None):
    source = Path(manifest["source_variants"][job["source_variant"]]["frozen_source_root"])
    data = manifest["versions"][job["version"]]
    checkpoint = smoke_target / "params" if smoke_target else Path(data["checkpoint_root"]) / job["tag"]
    config = resolve_path(str(checkpoint) + "_config.json")
    params = resolve_path(str(checkpoint) + ".safetensors")
    for path in (config, params):
        if not path.is_file():
            raise FileNotFoundError(path)
    prefix = smoke_target / "standalone_eval" if smoke_target else artifact_path(data["evaluation_root"]) / job["tag"]
    command = [PYTHON, "-u", str(source / "analysis/evaluate_partner_modelling.py"),
               "--config", str(config), "--params", str(params), "--layouts_dir", manifest["frozen_layouts_dir"],
               "--n_episodes_per_capability", "1" if smoke_target else "20", "--seed", "12345", "--out_prefix", str(prefix)]
    if job["condition"].startswith("rnn"):
        command.append("--save_hidden")
    return command, source, prefix


def evaluate_job(manifest, job, smoke_target=None):
    command, source, prefix = evaluation_command(manifest, job, smoke_target)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=source, env=environment(source, cpu=bool(smoke_target)), check=True)
    for extension in ("_summary.json", "_train.h5", "_test.h5"):
        if not Path(str(prefix) + extension).is_file():
            raise FileNotFoundError(str(prefix) + extension)
    if not smoke_target:
        publish(manifest["versions"][job["version"]]["evaluation_root"], job["tag"])
