"""Rebuild ignored schedule arrays from frozen sources and manifest parameters."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import struct
import sys
import uuid
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repo_paths import resolve_path, resolve_record_paths


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class OriginalZipInfo(zipfile.ZipInfo):
    def FileHeader(self, zip64=None):
        header = bytearray(super().FileHeader(zip64))
        if zip64:
            # Python 3.10 retains small sizes in forced ZIP64 local headers;
            # newer Python writes size sentinels and version 4.5 instead.
            self.create_version = max(self.create_version, 45)
            self.extract_version = max(self.extract_version, 45)
            struct.pack_into("<H", header, 4, self.extract_version)
            struct.pack_into("<II", header, 18, 0xffffffff, 0xffffffff)
        return bytes(header)


def save_original_archive(schedule, path):
    # NumPy 2 uses ZIP64 headers even for small members. NumPy 1 may not;
    # write that wrapper explicitly so recovery preserves the frozen checksum.
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for key in ("layouts", "profile_order", "capability_pairs", "seed",
                    "n_layouts", "passes_per_cycle"):
            member = OriginalZipInfo(f"{key}.npy")
            member.create_system = 3
            member.external_attr = 0o600 << 16
            with archive.open(member, "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asanyarray(getattr(schedule, key)),
                                          allow_pickle=False)
    return {**schedule.summary(), "sha256": sha(path)}


def rebuild(manifest_path, output_dir=None):
    manifest = resolve_record_paths(json.loads(resolve_path(manifest_path).read_text()))
    version = manifest["versions"]["v2"]
    source = Path(version["frozen_source_root"])
    scheduler = next((source / name for name in (
        "train/counterbalanced_scheduler.py", "baselines/IPPO/counterbalanced_scheduler.py")
        if (source / name).is_file()), None)
    if scheduler is None:
        raise FileNotFoundError("Frozen counterbalanced sampler is missing")
    population = source / "jaxmarl/environments/coordination_grid/capability_populations.py"
    for path in (scheduler, population):
        relative = str(path.relative_to(source))
        if sha(path) != version["source_sha256"][relative]:
            raise ValueError(f"Frozen source checksum mismatch: {relative}")
    spec = importlib.util.spec_from_file_location("_frozen_counterbalanced_sampler", scheduler)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    diverse_pairs = runpy.run_path(str(population))["TRAIN_CAPABILITY_PAIRS"]
    for regime, summary in manifest["schedules"].items():
        destination = (Path(output_dir) / f"{regime}.npz" if output_dir else
                       Path(manifest["schedule_paths"][regime]))
        expected = summary["sha256"]
        if destination.exists():
            if sha(destination) != expected:
                raise ValueError(f"Existing schedule checksum mismatch: {destination}")
            print(f"Verified existing {destination}", flush=True)
            continue
        pairs = diverse_pairs if regime == "diverse" else [(1, 4)]
        if len(pairs) != summary["n_profiles"]:
            raise ValueError("Profile count disagrees with manifest")
        print(f"Rebuilding {regime}: {summary['preallocated_episodes']:,} episodes", flush=True)
        schedule = module.build_schedule(pairs, summary["n_layouts"],
            summary["rounds_per_episode"], summary["preallocated_episodes"], summary["seed"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".rebuild-{regime}-{uuid.uuid4().hex}.npz")
        try:
            generated = save_original_archive(schedule, temporary)
            if generated != summary:
                raise ValueError(f"Regenerated parameters/checksum differ for {regime}; original files were preserved")
            temporary.replace(destination)
            metadata = destination.with_suffix(".json")
            if not metadata.exists():
                metadata.write_text(json.dumps(summary, indent=2) + "\n")
        finally:
            temporary.unlink(missing_ok=True)
            temporary.with_suffix(".json").unlink(missing_ok=True)
        print(f"Restored {destination}; SHA-256 matches the original run", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Optional destination for verification without touching original arrays")
    args = parser.parse_args()
    rebuild(args.manifest, args.output_dir)
