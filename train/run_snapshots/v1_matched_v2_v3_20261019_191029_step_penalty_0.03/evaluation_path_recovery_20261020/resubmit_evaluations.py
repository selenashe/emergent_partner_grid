"""Resume the failed evaluations and their dependent behavior/plotting stage."""
import os
from pathlib import Path
import subprocess
import sys


path = Path(sys.argv[1]).resolve()
root = path.parents[2]
os.environ['GRID_PROJECT_ROOT'] = str(root)
sys.path.insert(0, str(root))
from bash import submit_step_penalty_training as launcher

m = launcher.verify(path)
recovery = m['evaluation_recovery']
retry_id = Path(recovery['directory']).name
assert recovery['status'] in ('validated', 'submitting', 'submitted')
assert len(recovery['smoke_checks']) == 5
assert all(check['passed'] for check in recovery['smoke_checks'])
assert len(recovery['validated_inputs']) == 60
for inputs in recovery['validated_inputs']:
    for name in ('config', 'params'):
        assert launcher.sha(inputs[name]) == inputs[name + '_sha256']

evaluations = [j for data in m['versions'].values() for j in data['evaluation_jobs']]
assert len(evaluations) == 12
behavior = next(j for j in m['downstream_jobs'] if j['stage'] == 'behavior')
pending = [j for j in [*evaluations, behavior] if j.get('retry_id') != retry_id]
ids = [j['job_id'] for j in [*m['training_jobs'], *pending]]
accounting = subprocess.check_output([
    'sacct', '-S', '2026-10-19', '-j', ','.join(ids), '-X', '-n', '-P',
    '-o', 'JobID,State,ExitCode'], text=True)
states = {parts[0]: (parts[1], parts[2]) for line in accounting.splitlines()
          if len(parts := line.split('|')) >= 3}
assert all(states[j['job_id']] == ('COMPLETED', '0:0') for j in m['training_jobs'])
for job in pending:
    assert states[job['job_id']][0] in ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL')

clean = {k: v for k, v in os.environ.items()
         if not k.startswith('SLURM_') and k not in ('JAX_PLATFORMS', 'CUDA_VISIBLE_DEVICES', 'XLA_FLAGS')}
wrapper = str(Path(m['infrastructure_root']) / 'bash/run_counterbalanced_seed_extension.sh')
recovery.update(status='submitting', training_verified_completed=60)
launcher.write(path, m)


def resubmit(entry, dependencies=()):
    if entry.get('retry_id') == retry_id:
        return
    previous = {k: v for k, v in entry.items() if k != 'previous_attempts'}
    previous['slurm_state'], previous['exit_code'] = states[entry['job_id']]
    # Training finished successfully; its expired scheduler IDs are unnecessary.
    command = [arg for arg in entry['command']
               if not arg.startswith(('--dependency=', '--kill-on-invalid-dep='))]
    if dependencies:
        index = command.index(wrapper)
        command[index:index] = ['--dependency=afterok:' + ':'.join(dependencies),
                                '--kill-on-invalid-dep=yes']
    job_id = subprocess.check_output(command, cwd=root, env=clean, text=True).strip().split(';')[0]
    if not job_id.isdigit():
        raise RuntimeError(job_id)
    entry.setdefault('previous_attempts', []).append(previous)
    entry.update(job_id=job_id, command=command, retry_id=retry_id,
                 resubmitted_at_utc=launcher.now())
    recovery['resubmitted_jobs'].append(dict(job_id=job_id, previous_job_id=previous['job_id'],
                                           stage=entry.get('stage', 'eval')))
    launcher.write(path, m)
    print(entry.get('stage', 'eval'), entry.get('condition', ''), job_id, flush=True)


for data in m['versions'].values():
    for entry in data['evaluation_jobs']:
        resubmit(entry)
resubmit(behavior, [entry['job_id'] for entry in evaluations])
recovery.update(status='submitted', submitted_at_utc=launcher.now(),
                evaluation_job_ids=[entry['job_id'] for entry in evaluations],
                plotting_job_id=behavior['job_id'])
m['status'] = 'submitted'
launcher.write(path, m)
print('Resubmitted 12 evaluation jobs and the dependent behavior/plotting job; training was not resubmitted.', flush=True)
