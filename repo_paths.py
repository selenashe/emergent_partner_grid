"""Resolve paths in historical experiment records after repository relocation.

Saved configs, manifests, rollout attributes and frozen sources are immutable
evidence. Readers resolve their old paths in memory instead of rewriting them.
"""
import json
from pathlib import Path

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


def _archive_path(relative):
    for old, new in ARCHIVE_RELOCATIONS.items():
        try:
            suffix = relative.relative_to(old)
        except ValueError:
            continue
        return ROOT / new / suffix
    return ROOT / relative


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
        return path
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
