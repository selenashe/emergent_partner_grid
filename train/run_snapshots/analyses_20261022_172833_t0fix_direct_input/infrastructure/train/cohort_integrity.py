"""Verify the independently sealed direct-input scientific integrity contract.

The contract was created only after the original manifest, config-equivalence,
and trainer-transformation checks passed. It preserves those verified settings
and source hashes without requiring retired CNN experiment files at runtime.
"""
import hashlib
import json
from pathlib import Path

import yaml

from train.experiment import CONDITIONS, SEEDS, VERSIONS
from train.experiment_runtime import sha


def verify(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    contract_path = Path(manifest["integrity_contract"])
    if sha(contract_path) != manifest["integrity_contract_sha256"]:
        raise ValueError("Scientific integrity contract changed")
    contract = json.loads(contract_path.read_text())
    expected = {(v, c, s) for v in VERSIONS for c in CONDITIONS for s in SEEDS}
    identities = [(j["version"], j["condition"], j["seed"]) for j in manifest["training_jobs"]]
    if len(identities) != 60 or set(identities) != expected:
        raise ValueError("Expected exactly sixty distinct policies")
    if (manifest["seeds"] != list(SEEDS) or contract["seeds"] != list(SEEDS)
            or manifest["observation_settings"] != contract["observation_settings"]
            or manifest["observation_settings"] != dict(GRID_ENCODER="flat", INCLUDE_IS_T0=True)):
        raise ValueError("Wrong cohort or observation settings")
    if manifest["schedule_sha256"] != contract["schedule_sha256"]:
        raise ValueError("Scientific schedule contract changed")
    if set(manifest["schedule_paths"]) != set(contract["schedule_sha256"]):
        raise ValueError("Scientific schedule inputs missing")
    for name, path in manifest["schedule_paths"].items():
        if sha(path) != contract["schedule_sha256"][name]:
            raise ValueError("Schedule drift: " + name)
    if set(manifest["source_variants"]) != set(contract["source_sha256"]):
        raise ValueError("Protocol source variants changed")
    for name, variant in manifest["source_variants"].items():
        if variant["source_sha256"] != contract["source_sha256"][name]:
            raise ValueError("Scientific source contract changed: " + name)
        for rel, digest in variant["source_sha256"].items():
            if sha(Path(variant["frozen_source_root"]) / rel) != digest:
                raise ValueError("Source drift: " + rel)
    for rel, digest in manifest["infrastructure_sha256"].items():
        if sha(Path(manifest["infrastructure_root"]) / rel) != digest:
            raise ValueError("Frozen launcher drift: " + rel)
    digest = hashlib.sha256()
    layouts = sorted(Path(manifest["frozen_layouts_dir"]).glob("*.json"))
    if len(layouts) != 1096:
        raise ValueError("Expected 1096 shared layouts")
    for path in layouts:
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    if (digest.hexdigest() != contract["layout_files_sha256"]
            or manifest["layout_files_sha256"] != contract["layout_files_sha256"]):
        raise ValueError("Layout drift")
    for job in manifest["training_jobs"]:
        variant = ("v1_matched_influence" if job["version"] == "v1_matched"
                   and job["condition"] != "rnn_diverse_noinfluence" else job["version"])
        if job["source_variant"] != variant:
            raise ValueError("Wrong condition-specific protocol source: " + job["tag"])
        if job["tag"] != f"{job['condition']}_seed{job['seed']}_t0fix_direct_input":
            raise ValueError("Wrong direct-input checkpoint identity")
        path = Path(job["config_path"])
        if sha(path) != job["config_sha256"]:
            raise ValueError("Config bytes changed: " + job["tag"])
        config = yaml.safe_load(path.read_text())
        key = f"{job['version']}/{job['condition']}/seed{job['seed']}"
        scientific = {k: v for k, v in config.items() if k != "SAVE_PARAMS_PATH"}
        if scientific != contract["configs"][key]["scientific_settings"]:
            raise ValueError("Scientific config changed: " + job["tag"])
        output = Path(manifest["versions"][job["version"]]["checkpoint_root"]) / (job["tag"] + ".safetensors")
        if config["SAVE_PARAMS_PATH"] != str(output):
            raise ValueError("Checkpoint output differs: " + job["tag"])
        if (config["TOTAL_TIMESTEPS"] != 60000000
                or config["NUM_ENVS"] * config["NUM_STEPS"] * 915 != 59965440
                or config["ENV_KWARGS"]["step_penalty"] != .01
                or config.get("PILOT_UPDATES")):
            raise ValueError("Wrong full-budget direct-input configuration")
    return manifest
