"""Validate immutable sources and run all eight version/condition smoke jobs."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ast
import json
import os
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bash.submit_counterbalanced_training import (ROOT, V1_REVISION, CONDITIONS, verify_manifest,
    write, relocate_v1_imports)
from repo_paths import resolve_record_paths


def check_preserved_implementation(snapshot, version):
    # Audit guide:
    # Compare selected syntax trees in frozen/recovered sources to check that scheduling
    # patches retained the intended protocol and network behavior. Comments do not
    # appear in syntax trees. This guards against changing the scientific condition
    # while patching exposure bookkeeping.
    #
    relative = 'train/ippo_rnn_coordination_grid.py'
    old_relative = 'baselines/IPPO/ippo_rnn_coordination_grid.py'
    baseline = (subprocess.check_output(['git', 'show', f'{V1_REVISION}:{old_relative}'], cwd=ROOT).decode()
                if version == 'v1' else (ROOT / relative).read_text())
    if version == 'v1':
        baseline = relocate_v1_imports(baseline)
    current = (snapshot / relative).read_text()
    def nodes(text):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(text).body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    old, new = nodes(baseline), nodes(current)
    for name in old.keys() & new.keys():
        if name not in ['make_train', 'main']:
            assert old[name] == new[name], f'{version}: changed non-sampling function {name}'
    env = 'jaxmarl/environments/coordination_grid/coordination_grid.py'
    expected = (subprocess.check_output(['git', 'show', f'{V1_REVISION}:{env}'], cwd=ROOT)
                if version == 'v1' else (ROOT / env).read_bytes())
    assert (snapshot / env).read_bytes() == expected
    eval_relative = 'eval/evaluate_partner_modelling.py'
    expected_eval = (subprocess.check_output(['git', 'show',
        f'{V1_REVISION}:analysis/evaluate_partner_modelling.py'], cwd=ROOT).decode()
        if version == 'v1' else (ROOT / eval_relative).read_text())
    if version == 'v1':
        expected_eval = relocate_v1_imports(expected_eval)
    assert (snapshot / eval_relative).read_text() == expected_eval


def validate(path):
    # Audit guide:
    # Verify sources/schedules, run sampler/protocol regressions and bounded
    # train/evaluation smoke checks, and record preflight results in the manifest. This
    # is preparation validation, not sixty-million-step training or a scientific result.
    # It can write temporary/check artifacts but does not submit Slurm jobs.
    #
    m = resolve_record_paths(json.loads(path.read_text()))
    verify_manifest(m)
    output = ROOT / 'train/run_snapshots' / m['batch'] / 'preflight'
    output.mkdir(exist_ok=True)
    records = m['preflight'].get('smoke_runs', [])
    m['preflight'].update({'passed': False, 'smoke_runs': records})
    write(path, m)
    python = '/nlp/scr/jshe/miniconda3/envs/emergent_partner_model/bin/python'
    cores = ','.join(str(c) for c in sorted(os.sched_getaffinity(0))[:4])
    for version, data in m['versions'].items():
        source = Path(data['frozen_source_root'])
        check_preserved_implementation(source, version)
        subprocess.run(['bash', '-n', str(source / 'bash/train_final_experiment.sh'),
                        str(source / 'bash/eval_all_checkpoints.sh')], check=True)
        for condition in CONDITIONS:
            if any(r['version'] == version and r['condition'] == condition and r['passed'] for r in records):
                continue
            target = output / version / condition
            target.mkdir(parents=True, exist_ok=True)
            model = 'mlp' if condition.startswith('mlp') else 'rnn'
            regime = 'single' if 'single' in condition else 'diverse'
            influence = 'false' if 'noinfluence' in condition else 'true'
            checkpoint = target / 'params.safetensors'
            command = ['taskset', '-c', cores, python, '-u',
                'train/ippo_rnn_coordination_grid.py', 'SEED=1', 'NUM_SEEDS=1',
                f'MODEL_TYPE={model}', f'PARTNER_REGIME={regime}', f'INFLUENCE={influence}',
                'NUM_ENVS=4', 'NUM_STEPS=64', 'TOTAL_TIMESTEPS=256', 'UPDATE_EPOCHS=1',
                'NUM_MINIBATCHES=4', 'ENV_KWARGS.max_steps=3', 'EVAL_EPISODES_PER_CAPABILITY=1',
                f'SAVE_PARAMS_PATH={checkpoint}', f'hydra.run.dir={target / "hydra"}',
                '+STDOUT_LOG=true', '+STDOUT_SUMMARY=false']
            env = os.environ.copy()
            env.update({'JAX_PLATFORMS': 'cpu', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                        'PYTHONDONTWRITEBYTECODE': '1', 'HYDRA_FULL_ERROR': '1',
                        'PYTHONPATH': str(source), 'CUDA_VISIBLE_DEVICES': ''})
            log = target / 'smoke.log'
            print(f'[preflight] {version} {condition}', flush=True)
            with log.open('w') as handle:
                result = subprocess.run(command, cwd=source, env=env, stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode:
                print(log.read_text()[-12000:], flush=True)
                raise RuntimeError(f'Smoke failed: {log}')
            audit = json.loads((target / 'params_sampling_audit.json').read_text())
            assert audit['collected_environment_steps'] == 256
            assert audit['allocated_episode_count_difference'] <= 1
            assert sum(p['episodes_completed'] for p in audit['profiles']) == 4
            assert audit['rounds_completed'] == 84
            assert (target / 'params_eval.json').exists()
            records.append({'version': version, 'condition': condition, 'passed': True,
                            'collected_environment_steps': 256, 'completed_episodes': 4,
                            'completed_rounds': 84, 'log': str(log), 'audit': str(target / 'params_sampling_audit.json')})
            write(path, m)
            print(f'[preflight passed] {version} {condition}', flush=True)
    verify_manifest(m)
    m['preflight'].update({'passed': True, 'source_and_corpus_hashes_verified': True,
                          'network_and_environment_unchanged': True,
                          'training_and_evaluation_use_categorical_sampling': True,
                          'launcher_bash_syntax_verified': True,
                          'completed_at_utc': datetime.now(timezone.utc).isoformat()})
    write(path, m)
    print('[preflight] all eight version/condition smoke runs passed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    validate(parser.parse_args().manifest)
