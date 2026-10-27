"""Keep one A100 allocation while recovering policies and releasing their fits.

The scientific recovery implementation remains frozen in recovery_source_v2.
Each policy is isolated in a fresh process to preserve its frozen module imports.
Only validated cache completion releases that policy's pending CPU tasks.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def ready(directory):
    for label in ("train","test","additional"):
        path = directory / (label+".h5")
        audit_path = path.with_suffix(".json")
        if not path.exists() or not audit_path.exists(): return False
        audit = json.loads(audit_path.read_text())
        if audit.get("status") != "passed": return False
        if "archive_stats" in audit:
            stat = path.stat()
            if audit["archive_stats"] != dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns):
                raise RuntimeError("Changed validated recovery input: "+str(path))
    return True


def save(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record,indent=2)+"\n")
    temporary.replace(path)


def main(manifest):
    suite = json.loads(manifest.read_text()); root = Path(suite["root"])
    progress = dict(status="running",completed_policy_indices=[],released_fit_tasks=[],started_utc=datetime.now(timezone.utc).isoformat())
    path = root / "recovery_batch_progress.json"
    for index,record in enumerate(suite["policies"]):
        directory = root / "inputs" / record["version"] / f"{record['condition']}_seed{record['seed']}"
        if not ready(directory):
            subprocess.run([sys.executable,"-u",str(root / "recovery_source_v2/representation_probe_suite.py"),
                            "recover",str(manifest),"--index",str(index)],check=True)
        if not ready(directory): raise RuntimeError("Recovery did not validate policy "+str(index))
        for name,job in suite["jobs"].items():
            if not name.startswith("fit_"): continue
            task = job["job_id"]+"_"+str(index)
            state = subprocess.check_output(["squeue","-h","-j",task,"-o","%T"],text=True).strip()
            if state == "PENDING":
                subprocess.run(["scontrol","update","JobId="+task,"Dependency="],check=True)
                progress["released_fit_tasks"].append(task)
        progress["completed_policy_indices"].append(index)
        progress["last_update_utc"] = datetime.now(timezone.utc).isoformat()
        save(path,progress)
        print("[BATCH RECOVERY COMPLETE]",index,record["version"],record["condition"],record["seed"],flush=True)
    progress.update(status="completed",completed_utc=datetime.now(timezone.utc).isoformat())
    save(path,progress)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
