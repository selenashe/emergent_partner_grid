"""Install the maintained observation modules into new protocol snapshots.

Historical source directories stay immutable. Actor masks, environment dynamics,
PPO losses and recurrent replay remain those of the selected protocol.
"""
import ast
from pathlib import Path

PIPELINE_VERSION = "phase_input_v1"
TRAINER_FILE = "baselines/IPPO/ippo_rnn_coordination_grid.py"
DEFAULTS = {"GRID_ENCODER": "flat", "INCLUDE_IS_T0": True}


def _audit_executed_updates(source):
    """Give historical audit functions the executed budget without altering LR."""
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    replacements = []
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "write_audit" and len(node.args) >= 4):
            arg = node.args[3]
            if isinstance(arg, ast.Name) and arg.id == "config":
                replacements.append((offsets[arg.lineno - 1] + arg.col_offset,
                    offsets[arg.end_lineno - 1] + arg.end_col_offset))
    for begin, end in reversed(sorted(replacements)):
        source = source[:begin] + 'dict(config, NUM_UPDATES=config["RUN_UPDATES"])' + source[end:]
    return source


def limit_training(source):
    """Permit a short prefix while retaining the original optimizer schedule."""
    if 'config["RUN_UPDATES"]' in source:
        return _audit_executed_updates(source)
    tree = ast.parse(source)
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Subscript)
                              and isinstance(t.value, ast.Name) and t.value.id == "config"
                              and isinstance(t.slice, ast.Constant) and t.slice.value == "NUM_UPDATES"
                              for t in n.targets))
    lines = source.splitlines(keepends=True)
    lines.insert(assignment.end_lineno, '''    run_updates = config.get("PILOT_UPDATES") or config["NUM_UPDATES"]
    if not isinstance(run_updates, int) or not 1 <= run_updates <= config["NUM_UPDATES"]:
        raise ValueError("PILOT_UPDATES must be between 1 and the scheduled NUM_UPDATES")
    config["RUN_UPDATES"] = run_updates
''')
    source = "".join(lines)
    old = '_update_step, runner_state, None, config["NUM_UPDATES"]'
    if source.count(old) != 1:
        raise ValueError("Expected one outer PPO update scan")
    return _audit_executed_updates(source.replace(old, '_update_step, runner_state, None, config["RUN_UPDATES"]'))


def upgrade_trainer(source):
    """Use one encoder implementation across fixed allocation, v2 and v3."""
    canonical = Path(__file__).with_name("ippo_rnn_coordination_grid.py").read_text()
    start, end = "class ScannedRNN(", "class ActorCriticCommRNN("
    modules = canonical[canonical.index(start):canonical.index(end)]
    source = source[:source.index(start)] + modules + source[source.index(end):]
    # Preserve the protocol-specific masks in each actor class.
    old_start = "        encoder = CommObsEncoder("
    old_end = "        embedding = nn.LayerNorm()(embedding)"
    count = source.count(old_start)
    if count not in (0, 2):
        raise ValueError("Expected both RNN and MLP observation encoders")
    for _ in range(count):
        begin = source.index(old_start)
        finish = source.index(old_end, begin) + len(old_end)
        source = source[:begin] + "        embedding = encode_observation(obs, self.config, activation)" + source[finish:]
    if '    config.setdefault("INCLUDE_IS_T0", True)' not in source:
        source = source.replace("def make_train(config):\n", '''def make_train(config):
    config.setdefault("GRID_ENCODER", "cnn")
    config.setdefault("INCLUDE_IS_T0", True)
''', 1)
    source = limit_training(source)
    ast.parse(source)
    return source
