"""Validate the recurrent replay contract used by the matched experiment."""
import ast

TRAINER_FILE = "baselines/IPPO/ippo_rnn_coordination_grid.py"


def validate_recurrent_replay(source):
    """Require collection reset flags for replay and post-step done flags for GAE."""
    tree = ast.parse(source)
    transition = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Name) and n.func.id == 'Transition')
    flags = {k.arg: k.value.id for k in transition.keywords
             if k.arg in {'reset', 'done'} and isinstance(k.value, ast.Name)}
    if flags != {'reset': 'last_done', 'done': 'done_all'}:
        raise ValueError('Matched collection must store separate replay reset and GAE done flags')
    if '(traj_batch.obs, traj_batch.reset),' not in source:
        raise ValueError('Matched PPO replay must use the pre-observation reset flag')
    return True
