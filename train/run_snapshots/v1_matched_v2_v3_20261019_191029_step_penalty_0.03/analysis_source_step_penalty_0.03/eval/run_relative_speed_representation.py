"""Prepare and run binary relative-speed probes on original specialist rollouts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(os.environ.get("GRID_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
from repo_paths import resolve_path

INPUT_MANIFEST = ROOT / "eval/relative_speed_inputs.json"
CONDITIONS = ("rnn_diverse_influence", "rnn_single_influence", "rnn_diverse_noinfluence")
VERSIONS = ("v1_matched", "v2", "v3")
ANALYSIS_PYTHON = "/juice6/u/nlp/climblab/tmp/epg_repr_all10_20261008/bin/python"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_inputs(manifest):
    expected = {(v, c, s) for v in VERSIONS for c in CONDITIONS for s in range(1, 6)}
    identities = [(r["version"], r["condition"], r["seed"]) for r in manifest["policies"]]
    if len(identities) != 45 or set(identities) != expected:
        raise ValueError("Require all 90 distinct specialist-policy input records")
    for record in manifest["policies"]:
        for field in ("checkpoint", "config"):
            if sha(resolve_path(record[field])) != record[field + "_sha256"]:
                raise ValueError(f"Saved policy input changed: {record[field]}")
        evaluator = resolve_path(record["source_root"]) / "analysis/evaluate_partner_modelling.py"
        if sha(evaluator) != record["evaluator_sha256"]:
            raise ValueError("Frozen evaluator changed")
        if not resolve_path(record["layouts_dir"]).is_dir():
            raise FileNotFoundError(record["layouts_dir"])
        if set(record["original_rollouts"]) != {"train", "test"}:
            raise ValueError("Only original specialist train/test rollout files are permitted")
        for population, path in record["original_rollouts"].items():
            stat = Path(path).stat()
            saved = record["original_stats"][population]
            if (stat.st_size, stat.st_mtime_ns) != (saved["size"], saved["mtime_ns"]):
                raise ValueError(f"Original specialist input changed: {path}")


def prepare(out, input_manifest=None):
    out = Path(out).resolve()
    if out.exists():
        raise ValueError("Use a fresh output root")
    input_manifest = Path(input_manifest or INPUT_MANIFEST).resolve()
    inputs = json.loads(input_manifest.read_text())
    verify_inputs(inputs)
    records = []
    fields = ("version", "condition", "seed", "source_root", "evaluator_sha256", "config", "config_sha256",
              "checkpoint", "checkpoint_sha256", "layouts_dir", "allocation_protocol", "original_rollouts", "original_stats")
    for previous in inputs["policies"]:
        record = {key: previous[key] for key in fields}
        stem = f"{record['condition']}_seed{record['seed']}"
        record["analysis_dir"] = str(out / record["version"] / "policies" / stem)
        records.append(record)
    out.mkdir(parents=True)
    (out / "jobs").mkdir()
    (out / "slurm_logs").mkdir()
    hashes = {}
    for relative in ("eval/run_relative_speed_representation.py", "eval/relative_speed_representation.py",
                     "eval/representation_analysis.py", "repo_paths.py",
                     "jaxmarl/environments/coordination_grid/capability_populations.py"):
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        hashes[relative] = sha(target)
    manifest = dict(status="prepared", schema_version=2, root=str(out), repo_root=str(ROOT),
        input_manifest=str(input_manifest), input_manifest_sha256=sha(input_manifest),
        prepared_utc=datetime.now(timezone.utc).isoformat(), policies=records, n_policies=45,
        versions=list(VERSIONS), conditions=list(CONDITIONS), seeds=list(range(1,6)),
        profiles_per_policy=46, episodes_per_profile=20, episodes_per_policy=920,
        train_episodes_per_policy=736, test_episodes_per_policy=184,
        train_counts=[368,368], test_counts=[92,92], classes={"0":"red faster","1":"blue faster"},
        analysis_python=ANALYSIS_PYTHON, source_sha256=hashes, jobs=[])
    write(out / "analysis_run.json", manifest)
    (out / "README.md").write_text("# Specialist-only binary representation analysis\n\nStatus: prepared. "
        "Original 46 profiles only, 920 episodes per policy, 736 fitting/184 held out. "
        "Classes: red faster and blue faster. No RL retraining or additional evaluations.\n")
    return out / "analysis_run.json"


def verify(manifest):
    root = Path(manifest["root"])
    for relative, digest in manifest["source_sha256"].items():
        if sha(root / "source" / relative) != digest:
            raise ValueError(f"Frozen analysis source changed: {relative}")
    verify_inputs(manifest)


def script(manifest, stage):
    if stage not in ("analyze", "aggregate"):
        raise ValueError("Only analysis and aggregation stages are supported")
    root = Path(manifest["root"])
    lines = ["#!/bin/bash", "set -euo pipefail", f"cd {shlex.quote(manifest['repo_root'])}",
        f"export GRID_PROJECT_ROOT={shlex.quote(manifest['repo_root'])}",
        "export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1",
        f"export MPLCONFIGDIR={shlex.quote(str(root/'mpl_cache'))}",
        f"export NUMBA_CACHE_DIR={shlex.quote(str(root/'numba_cache'))}"]
    command = [manifest["analysis_python"], str(root / "source/eval/relative_speed_representation.py"), stage,
               str(root / "analysis_run.json")]
    lines.append(shlex.join(command) + (" ${SLURM_ARRAY_TASK_ID:-0}" if stage == "analyze" else ""))
    path = root / "jobs" / f"{stage}.sh"
    path.write_text("\n".join(lines) + "\n")
    subprocess.run(["bash", "-n", str(path)], check=True)
    return path


def submit(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    verify(manifest)
    if manifest["status"] == "completed":
        raise ValueError("Completed run")
    root = Path(manifest["root"])
    for stage in ("analyze", "aggregate"):
        if any(job["stage"] == stage for job in manifest["jobs"]):
            continue
        command = ["sbatch", "--parsable", "--account=nlp", "--partition=sphinx", "--cpus-per-task=2",
            "--mem=16G", "--time=01:00:00", f"--job-name=repr_binary_{stage}",
            f"--output={root}/slurm_logs/{stage}_%A_%a.out", f"--error={root}/slurm_logs/{stage}_%A_%a.err"]
        if stage == "analyze":
            command += ["--array=0-89%10"]
        else:
            dependency = next(job["job_id"] for job in manifest["jobs"] if job["stage"] == "analyze")
            command += [f"--dependency=afterok:{dependency}", "--kill-on-invalid-dep=yes"]
        job_script = script(manifest, stage)
        result = subprocess.run(command + [str(job_script)], check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            raise ValueError(result.stdout)
        manifest["jobs"].append(dict(stage=stage, job_id=job_id, script=str(job_script)))
        manifest["status"] = "submitted"
        write(path, manifest)
        print(stage, job_id, flush=True)


def local(path, cpus):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    verify(manifest)
    root = Path(manifest["root"])
    if manifest["jobs"]:
        ids = ",".join(job["job_id"] for job in manifest["jobs"])
        active = subprocess.run(["squeue","-h","-j",ids,"-o","%i"], check=True, capture_output=True, text=True).stdout.strip()
        if active:
            raise ValueError("Submitted jobs are still active")
    env = dict(os.environ, GRID_PROJECT_ROOT=manifest["repo_root"], PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
        MPLCONFIGDIR=str(root/'mpl_cache'), NUMBA_CACHE_DIR=str(root/'numba_cache'))
    manifest["status"] = "running_local"
    manifest["local_execution"] = dict(hostname=os.uname().nodename, allocation_job_id=os.environ.get("SLURM_JOB_ID"),
        started_utc=datetime.now(timezone.utc).isoformat(), processes=[])
    write(path, manifest)
    processes, handles = [], []
    try:
        for version, cpu in zip(VERSIONS, cpus):
            log = root / "slurm_logs" / f"local_analyze_{version}.log"
            handle = log.open("a"); handles.append(handle)
            command = ["taskset","-c",str(cpu),manifest["analysis_python"],str(root/'source/eval/relative_speed_representation.py'),
                       "analyze-version",str(path),"--version",version]
            process = subprocess.Popen(command, env=env, cwd=manifest["repo_root"], stdout=handle, stderr=subprocess.STDOUT)
            processes.append(process)
            manifest["local_execution"]["processes"].append(dict(version=version,pid=process.pid,cpu_id=cpu,log=str(log)))
            write(path,manifest)
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):
                raise RuntimeError("A specialist analysis worker failed; inspect logs")
            done=sum((Path(r["analysis_dir"])/'completion.json').exists() for r in manifest['policies'])
            print(f"[progress] specialist analyses {done}/45",flush=True)
            time.sleep(20)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("A specialist analysis worker failed")
        subprocess.run([manifest["analysis_python"],str(root/'source/eval/relative_speed_representation.py'),
                        "aggregate",str(path)],env=env,cwd=manifest["repo_root"],check=True)
    except BaseException:
        for process in processes:
            if process.poll() is None:process.terminate()
        manifest['status']='failed_local';write(path,manifest)
        raise
    finally:
        for handle in handles:handle.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','submit','local'))
    parser.add_argument('path',type=Path)
    parser.add_argument('--input-manifest',type=Path)
    parser.add_argument('--cpus',type=int,nargs='+',help='Three analysis-worker CPUs within allowed affinity')
    args=parser.parse_args()
    if args.mode=='prepare':print(prepare(args.path,args.input_manifest))
    elif args.mode=='submit':submit(args.path)
    else:
        if not args.cpus or len(args.cpus)!=3 or not set(args.cpus)<=os.sched_getaffinity(0):
            parser.error('local requires three allowed CPU assignments')
        local(args.path,args.cpus)


if __name__=='__main__':main()
