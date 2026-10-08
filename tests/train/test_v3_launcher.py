"""V3 submissions preserve dependencies and resume without duplicating work."""
from pathlib import Path

from bash import submit_v3_counterbalanced_training as launcher


def test_submission_guards_40_matched_policies_and_queues_evaluation_and_plots(tmp_path, monkeypatch):
    m = dict(status='prepared', preflight={'passed': True},
             infrastructure_root=str(tmp_path / 'infrastructure'), log_dir=str(tmp_path / 'logs'),
             versions={'v3': {'evaluation_jobs': []}}, aggregation_jobs=[],
             training_jobs=[dict(version='v3', condition=c, seed=s, tag=f'{c}_seed{s}')
                            for c in launcher.CONDITIONS for s in launcher.SEEDS])
    commands = []
    def capture(command, **kwargs):
        commands.append(command)
        return str(1000 + len(commands)) + '\n'
    monkeypatch.setattr(launcher, 'verify', lambda _: m)
    monkeypatch.setattr(launcher, 'write', lambda *args: None)
    monkeypatch.setattr(launcher.subprocess, 'check_output', capture)
    path = tmp_path / 'manifest.json'
    launcher.submit(path)
    assert len(commands) == 47  # two GPU checks, forty policies, four evaluations, one plot job
    wrapper = str(tmp_path / 'infrastructure/bash/run_counterbalanced_seed_extension.sh')
    for command in commands:
        index = command.index(wrapper)
        assert Path(command[index + 1]).is_absolute()
        assert command[index + 1].endswith('submit_v3_counterbalanced_training.py')
        stage = command[index + 2]
        if stage == 'train':
            assert '--dependency=afterok:1001:1002' in command
            assert '--time=08:00:00' in command
        elif stage == 'eval':
            dependencies = next(a for a in command if a.startswith('--dependency=')).split(':')[1:]
            condition = command[-1]
            assert dependencies == [j['job_id'] for j in m['training_jobs'] if j['condition'] == condition]
        elif stage == 'aggregate':
            assert '--gres=gpu:1' not in command
            assert '--dependency=afterok:1043:1044:1045:1046' in command
        if stage != 'production-smoke':
            assert '--kill-on-invalid-dep=yes' in command
    launcher.submit(path)
    assert len(commands) == 47
