"""Verify all fitted archives, align plot timing captions, then release plots."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def write(path,value):
    tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(value,indent=2)+"\n");tmp.replace(path)


def caption(code,previous_memory):
    code=code.replace("Action following h_t assigns faster goal","Actual faster-goal assignment after observation t")
    old="Prior-event curves include transitions strictly before h_t; action curve follows h_t. Gray windows are descriptive maxima."
    new="Events: transitions before observation t. Assignment: after observation t. Gray: descriptive maxima."
    if previous_memory:new+="\\nPrevious-memory decoding excludes observation t."
    if old not in code and new not in code:raise ValueError("Unexpected behavior figure caption")
    return code.replace(old,new)


def main(manifest):
    suite=json.loads(manifest.read_text());root=Path(suite["root"]);base=Path(suite["baseline_root"])
    if digest(base/"analysis_run.json")!=suite["baseline_manifest_sha256"]:raise ValueError("Original baseline manifest changed")
    original=json.loads((base/"analysis_run.json").read_text())
    for record in original["policies"]:
        directory=Path(record["probe_directory"]);done=json.loads((directory/"completion.json").read_text())
        if digest(directory/"training_capability_probes.npz")!=done["archive_sha256"]:raise ValueError("Original fitted archive changed")
    checked=0
    for name in suite["representations"]:
        directory=root/name;path=directory/"analysis_run.json";run=json.loads(path.read_text())
        for record in run["policies"]:
            policy=Path(record["probe_directory"]);done=json.loads((policy/"completion.json").read_text())
            if done["status"]!="completed" or done["feature_dimension"]!=run["feature_dimension"]:raise ValueError("Incomplete or incorrectly shaped readout")
            if digest(policy/"training_capability_probes.npz")!=done["archive_sha256"]:raise ValueError("Fitted archive hash mismatch")
            checked+=1
        source=directory/"source/analyze_decoding_behavior.py"
        old=source.read_text();new=caption(old,name=="previous_memory")
        if new!=old:
            history=directory/"source_history/pre_caption_alignment";history.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,history/source.name)
            write(history/"source_sha256.json",run["source_sha256"])
            source.write_text(new);py_compile.compile(str(source),doraise=True)
            run.setdefault("source_revisions",[]).append(dict(reason="Caption-only clarification of global observation timing; fitted features and classifier code unchanged.",
                prior_hash=digest(history/source.name),file=source.name,created_utc=datetime.now(timezone.utc).isoformat()))
            run["source_sha256"][source.name]=digest(source)
            write(path,run)
        (directory/"REPRESENTATION_TIMING.md").write_text("# Representation timing\n\n"
            "Every curve is indexed by the current observation t. Events describe transitions before that observation; actual assignments describe transition t. "
            "Previous-memory inputs are h_(t-1) and therefore exclude observation t. The other features use observation t or the policy's updated h_t. "
            "Previous memory at the final valid observation L-1 is h_(L-2), which is held forward after termination.\n")
    if checked!=240:raise ValueError("Expected240 completed policy/representation fits")
    audit=dict(status="passed",completed_utc=datetime.now(timezone.utc).isoformat(),completed_fits=checked,
               all_fitted_archive_hashes_verified=True,original_baseline_manifest_and_30_archives_unchanged=True,
               captions_use_global_observation_index=True,previous_memory_timing_explicit=True,
               producer_sha256=digest(Path(__file__)))
    write(root/"pre_aggregation_audit.json",audit)
    subprocess.run(["scontrol","release",suite["jobs"]["aggregate"]["job_id"]],check=True)
    current=json.loads(manifest.read_text());current.pop("aggregation_hold_reason",None);current["status"]="aggregating";write(manifest,current)
    print("PASS:240 fitted archives verified; original baseline unchanged; representation-specific plots released.",flush=True)


if __name__=="__main__":main(Path(sys.argv[1]).resolve())
