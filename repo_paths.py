"""Resolve paths in historical experiment records after repository relocation.

Saved configs, manifests, rollout attributes and frozen sources are immutable
evidence. Readers resolve their old paths in memory instead of rewriting them.
"""
import json
import os
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parent
ORIGINAL_ROOT = Path("/juice6/u/jshe/emergent_partner_grid")
ARCHIVE_INDEX = ROOT / "archive/relocations.json"
ARCHIVE_RELOCATIONS = (json.loads(ARCHIVE_INDEX.read_text())
                       if ARCHIVE_INDEX.exists() else {})
RELOCATIONS = (
    ("dev/grids_capability_selected_balanced_1096", "data_prep/grids_capability_selected_balanced_1096"),
    ("dev/grids_capability_selected_2000", "data_prep/grids_capability_selected_2000"),
    ("dev/train_logs", "train/train_logs"),
    ("dev/eval_out", "eval/eval_out"),
    ("dev/run_snapshots", "train/run_snapshots"),
    ("analysis/training_curves", "train/training_curves"),
    ("analysis/protocol_comparison", "eval/protocol_comparison"),
    ("analysis/representation_results", "eval/representation_results"),
    ("outputs", "train/hydra_outputs"),
)


def artifact_root():
    """Shared storage for weights, rollouts and numerical training artifacts."""
    root = Path(os.environ.get(
        "GRID_ARTIFACT_ROOT", "/juice6/scr6/nlp/interp-models/emergent_partner_grid"
    )).expanduser()
    if not root.is_absolute():
        raise ValueError("GRID_ARTIFACT_ROOT must be an absolute path")
    return root


def artifact_path(value):
    """Map a project path to scratch, preserving its repository-relative tree.

    Explicit paths outside the project (including temporary test outputs) stay
    unchanged. Historical dev/analysis paths use the same relocation rules as
    readers. This function also maps directories for frozen training launchers.
    """
    path = Path(value)
    relative = path
    if path.is_absolute():
        for base in (ROOT, ORIGINAL_ROOT):
            try:
                relative = path.relative_to(base)
                break
            except ValueError:
                pass
        else:
            return path
    if ".." in relative.parts:
        raise ValueError("Artifact paths must stay within the project")
    for old, new in RELOCATIONS:
        try:
            relative = Path(new) / relative.relative_to(old)
            break
        except ValueError:
            pass
    for old, new in ARCHIVE_RELOCATIONS.items():
        try:
            relative = Path(new) / relative.relative_to(old)
            break
        except ValueError:
            pass
    return artifact_root() / relative


def prepare_artifact_path(value):
    """Create the scratch parent before writing a binary artifact."""
    path = artifact_path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def link_artifact(value):
    """Publish a compatibility link after a binary was successfully written."""
    logical = Path(value)
    if not logical.is_absolute():
        logical = ROOT / logical
    target = artifact_path(value)
    if logical == target:
        return target
    if not target.is_file():
        raise FileNotFoundError(target)
    logical.parent.mkdir(parents=True, exist_ok=True)
    temporary = logical.with_name(logical.name + ".link-" + uuid.uuid4().hex)
    try:
        temporary.symlink_to(target)
        temporary.replace(logical)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def save_numpy_archive(value, *, compressed=False, **arrays):
    """Save binary arrays on scratch while keeping logical read paths valid."""
    import numpy as np
    path = prepare_artifact_path(value)
    (np.savez_compressed if compressed else np.savez)(path, **arrays)
    link_artifact(value)


def _existing_artifact(path):
    if path.is_symlink():
        return path.resolve()
    if path.exists():
        return path
    candidate = artifact_path(path)
    return candidate if candidate.exists() else path


def _archive_path(relative):
    for old, new in ARCHIVE_RELOCATIONS.items():
        try:
            suffix = relative.relative_to(old)
        except ValueError:
            continue
        return _existing_artifact(ROOT / new / suffix)
    return _existing_artifact(ROOT / relative)


def resolve_path(value):
    """Return an existing relocated path, leaving unrelated paths untouched."""
    # Audit guide:
    # Translate historical experiment paths to current active or archived locations only
    # for reading. Prefer a path that already exists, and leave unrelated external paths
    # alone. Preserve original files/config values so their hashes and historical
    # evidence remain intact.
    #
    path = Path(value)
    if path.exists():
        return path.resolve() if path.is_symlink() else path
    relative = path
    if path.is_absolute():
        for base in (ROOT, ORIGINAL_ROOT):
            try:
                relative = path.relative_to(base)
                break
            except ValueError:
                pass
        else:
            return path
    if relative.parts and relative.parts[0] == "logs":
        name = relative.name
        directory = "train/manifests" if name.startswith("sbatch_") else (
            "eval/slurm_logs" if name.startswith("cg_eval") else "train/slurm_logs")
        return _archive_path(Path(directory) / name)
    if (relative.name in {"sampling_verification.json", "actual_exposure_by_run.csv", "actual_exposure_by_profile.csv"}
            and len(relative.parts) >= 4 and relative.parts[0] in {"analysis", "eval"}
            and relative.parts[1] == "protocol_comparison"):
        return _archive_path(Path("train/sampling_audits") / relative.parts[2] / relative.name)
    for old, new in RELOCATIONS:
        try:
            suffix = relative.relative_to(old)
        except ValueError:
            continue
        return _archive_path(Path(new) / suffix)
    relocated = _archive_path(relative)
    return relocated if relocated.exists() else path


def resolve_record_paths(record):
    """Resolve recorded path strings without mutating the original record."""
    # Audit guide:
    # Return a recursively reconstructed record whose repository path strings resolve to
    # present locations. Do not mutate the original manifest. This lets current readers
    # consume immutable records that refer to retired dev/analysis/logs locations.
    #
    if isinstance(record, dict):
        return {key: resolve_record_paths(value) for key, value in record.items()}
    if isinstance(record, list):
        return [resolve_record_paths(value) for value in record]
    if isinstance(record, str) and (record.startswith(str(ORIGINAL_ROOT) + "/")
                                   or record.startswith(str(ROOT) + "/")):
        return str(resolve_path(record))
    return record
