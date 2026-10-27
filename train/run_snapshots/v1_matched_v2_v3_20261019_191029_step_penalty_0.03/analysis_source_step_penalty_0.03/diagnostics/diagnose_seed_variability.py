"""Retrospective PPO seed audit; never changes training sources or policies.

Uses the 120 manifest-mapped training histories and final policy evaluations.
Run with NumPy and Matplotlib. Add --rollouts in an environment with h5py to
measure assignment dwell times and replay the scripted partner's delay clock.
"""

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repo_paths import resolve_path

VERSIONS = ("v1_matched", "v2", "v3")
CONDITIONS = ("rnn_diverse_influence", "mlp_diverse_influence",
              "rnn_single_influence", "rnn_diverse_noinfluence")
LABELS = ("Diverse RNN + influence", "Diverse MLP + influence",
          "Single RNN + influence", "Diverse RNN, no influence")
HISTORY = ROOT / "/juice6/u/jshe/emergent_partner_grid/train/training_curves/v1_matched_v2_v3_20261019_191029_step_penalty_0.03"
EVALUATION = ROOT / "/juice6/u/jshe/emergent_partner_grid/eval/protocol_comparison/v1_matched_v2_v3_20261019_191029_step_penalty_0.03"
PROBES = ROOT / "/juice6/u/jshe/emergent_partner_grid/eval/representation_results/relative_speed_v1_matched_v2_v3_20261019_191029_step_penalty_0.03/probe_per_seed.csv"
WINDOWS = ((1, 46), (47, 92), (93, 183), (184, 366), (367, 549),
           (550, 732), (733, 869), (870, 915))
SOURCES = [
    ("PPO implementation and recurrent debugging checks",
     "https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/"),
    ("What Matters In On-Policy Reinforcement Learning?",
     "https://arxiv.org/abs/2006.05990"),
    ("Implementation Matters in Deep Policy Gradients",
     "https://arxiv.org/abs/2005.12729"),
    ("Deep Reinforcement Learning that Matters",
     "https://arxiv.org/abs/1709.06560"),
    ("The Primacy Bias in Deep Reinforcement Learning (off-policy evidence)",
     "https://arxiv.org/abs/2205.07802"),
]


def read_csv(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def identity(row):
    return row["version"], row["condition"], int(row["seed"])


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary_series(content, names):
    for name in names:
        match = re.search(r"^\s+" + name + r": \[(.*)\]$", content, re.M)
        if match:
            values = np.array([float(x) for x in match[1].split(",")])
            if len(values) != 915 or not np.isfinite(values).all():
                raise ValueError(f"Invalid {name} series")
            return values
    return np.full(915, np.nan)


def trailing(values, width=46):
    # Full windows only: do not mistake no-completed-episode placeholders for 0.
    valid = np.isfinite(values)
    sums = np.convolve(np.where(valid, values, 0), np.ones(width), "valid")
    counts = np.convolve(valid.astype(float), np.ones(width), "valid")
    return np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts == width)


def rank(values):
    # Average tied ranks; no SciPy dependency needed.
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def correlation(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def audit_histories(out):
    metadata = json.loads((HISTORY / "summary.json").read_text())
    updates = {}
    for row in read_csv(HISTORY / "updates.csv"):
        updates.setdefault(identity(row), []).append(row)
    expected = {(v, c, s) for v in VERSIONS for c in CONDITIONS for s in range(1, 6)}
    if set(updates) != expected or len(metadata["runs"]) != 120:
        raise ValueError("Need all 120 policies")
    behavior = {identity(r): r for r in read_csv(
        EVALUATION / "assignment_behaviors_all_seeds/novel/per_seed_summary.csv")}
    evaluation = {pop: {identity(r): r for r in read_csv(
        EVALUATION / f"all_5_seeds_step_penalty_0.03/{pop}/per_seed.csv") if r["version"] in VERSIONS}
        for pop in ("familiar", "novel")}
    probes = {(r["version"], r["condition"], int(r["training_seed"])): r
              for r in read_csv(PROBES) if r["axis"] == "round" and r["cutoff"] == "20"
              and r["kind"] == "primary" and r["population"] == "all"}
    rows, windows, curves, provenance = [], [], {}, []
    for record in metadata["runs"]:
        key = identity(record)
        raw = updates[key]
        if len(raw) != 915 or [int(r["update"]) for r in raw] != list(range(1, 916)):
            raise ValueError(f"Incomplete history: {key}")
        data = {k: np.array([float(r[k]) for r in raw]) for k in raw[0]
                if k not in ("version", "condition", "seed")}
        log = resolve_path(record["log"])
        if sha256(log) != record["log_sha256"]:
            raise ValueError(f"Training log changed: {key}")
        content = log.read_text()
        data["initial_red"] = summary_series(content, ("initial_alloc_red", "t0_alloc_red"))
        data["online_red"] = summary_series(content, ("online_alloc_red",))
        curves[key] = data
        config_path = resolve_path(record["config"])
        config = json.loads(config_path.read_text())
        if sha256(config_path) != record["config_sha256"]:
            raise ValueError(f"Configuration changed: {key}")
        audit_path = resolve_path(record["sampling_audit"])
        audit = json.loads(audit_path.read_text())
        if int(data["rounds_completed"].sum()) != audit["rounds_completed"]:
            raise ValueError(f"Exposure audit mismatch: {key}")
        smoothed = trailing(data["episode_return"])
        best_index = int(np.nanargmax(smoothed))
        running_best = np.maximum.accumulate(np.where(np.isfinite(smoothed), smoothed, -np.inf))
        drawdown = running_best - smoothed
        drawdown_index = int(np.nanargmax(drawdown))
        row = dict(version=key[0], condition=key[1], seed=key[2],
                   final46_training_return=float(smoothed[-1]),
                   best46_training_return=float(smoothed[best_index]),
                   best46_end_update=best_index + 46,
                   best46_end_million_steps=(best_index + 46) * 65536 / 1e6,
                   best_to_final_drop=float(smoothed[best_index] - smoothed[-1]),
                   maximum46_drawdown=float(drawdown[drawdown_index]),
                   maximum46_drawdown_end_update=drawdown_index + 46,
                   completed_training_rounds=audit["rounds_completed"],
                   final46_initial_red=float(data["initial_red"][-46:].mean()),
                   final46_online_red=float(data["online_red"][-46:].mean()),
                   final46_entropy_t0=float(data["entropy_t0"][-46:].mean()),
                   final46_entropy_movement=float(data["entropy_movement"][-46:].mean()),
                   final46_value_loss=float(data["value_loss"][-46:].mean()))
        for lo, hi in WINDOWS:
            window = dict(version=key[0], condition=key[1], seed=key[2],
                          first_update=lo, last_update=hi,
                          million_steps_start=(lo - 1) * 65536 / 1e6,
                          million_steps_end=hi * 65536 / 1e6)
            for name in ("episode_return", "round_success", "episode_steps", "entropy_t0",
                         "entropy_movement", "assignment_switch_rate", "initial_red", "online_red"):
                x = data[name][lo - 1:hi]
                window[name] = float(np.nanmean(x)) if np.isfinite(x).any() else None
            windows.append(window)
            row[f"return_updates_{lo}_{hi}"] = window["episode_return"]
        for pop in evaluation:
            e = evaluation[pop][key]
            for name in ("episode_return", "episode_steps", "success", "switches_per_round"):
                row[pop + "_" + name] = float(e[name])
            # Exact identity from +1 success, -.01 otherwise, terminal included.
            reconstructed = 1.01 * 20 * float(e["success"]) - .01 * float(e["episode_steps"])
            row[pop + "_return_identity_error"] = float(e["episode_return"]) - reconstructed
            if abs(row[pop + "_return_identity_error"]) > 2e-4:
                raise ValueError(f"Reward identity failed: {key}, {pop}")
        for name in ("initial_ego_red", "final_ego_red", "fraction_rounds_switched",
                     "final_partner_faster", "final_partner_faster_round1", "final_partner_faster_round20"):
            row["novel_" + name] = float(behavior[key][name])
        for name in ("balanced_accuracy", "recall_0", "recall_1"):
            row["relative_speed_probe_round20_" + name] = float(probes[key][name]) if key in probes else None
        rows.append(row)
        provenance.append({**record, "sampling_audit_sha256": sha256(audit_path),
                           "rounds_completed": audit["rounds_completed"],
                           "ppo": {k: config[k] for k in ("LR", "LR_WARMUP", "GAMMA", "GAE_LAMBDA",
                               "NUM_STEPS", "NUM_ENVS", "NUM_MINIBATCHES", "UPDATE_EPOCHS", "ENT_COEF")}})
    correlations = []
    for version in VERSIONS:
        for condition in CONDITIONS:
            group = [r for r in rows if (r["version"], r["condition"]) == (version, condition)]
            final = [r["final46_training_return"] for r in group]
            for lo, hi in WINDOWS[:-1]:
                early = [r[f"return_updates_{lo}_{hi}"] for r in group]
                correlations.append(dict(version=version, condition=condition, first_update=lo,
                    last_update=hi, n=10, pearson=correlation(early, final),
                    spearman=correlation(rank(early), rank(final))))
    rows.sort(key=identity)
    write_csv(out / "per_seed.csv", rows)
    write_csv(out / "training_windows.csv", windows)
    write_csv(out / "early_final_correlations.csv", correlations)
    return rows, curves, provenance


def audit_rollouts(out):
    import h5py
    metadata = json.loads((EVALUATION / "assignment_behaviors_all_seeds/novel/analysis.json").read_text())
    records = metadata["sources"]
    rows = []
    for record in records:
        version, condition, seed = identity(record)
            # Include all five seeds in every condition to separate failure modes.
        for population in ("train", "test"):
            path = resolve_path(record["source"]["path"])
            path = path.with_name(path.name.replace("_test.h5", f"_{population}.h5"))
            with h5py.File(path) as f:
                a = {k: np.asarray(f[k]) for k in ("dones", "round_done", "round_idx", "is_t0",
                    "round_time", "partner_assignment", "success", "rewards")}
                profile = np.asarray(f["capability_index_per_ep"])
                pool = np.asarray(f["capability_pool"])
            if not a["dones"].any(axis=1).all():
                raise ValueError("Missing episode termination")
            alive = np.arange(a["dones"].shape[1])[None, :] <= a["dones"].argmax(axis=1)[:, None]
            endings = a["round_done"] & alive
            if not (endings.sum(axis=1) == 20).all():
                raise ValueError("Incomplete episode")
            switches = np.zeros_like(alive)
            switches[:, 1:] = ((a["partner_assignment"][:, 1:] != a["partner_assignment"][:, :-1])
                & (a["round_idx"][:, 1:] == a["round_idx"][:, :-1]) & alive[:, 1:])
            # Replay only the exact partner delay clock, independent of navigation.
            # A switch reloads the new delay BEFORE movement; t0 never moves.
            counter = np.zeros(len(profile), int)
            denied = np.zeros_like(alive)
            eligible = np.zeros_like(alive)
            for tick in range(alive.shape[1]):
                initial = a["is_t0"][:, tick]
                counter[initial] = 0
                partner_goal = 2 - a["partner_assignment"][:, tick]  # RED=0, BLUE=1
                delay = pool[profile, np.clip(partner_goal, 0, 1)]
                movement = alive[:, tick] & ~initial
                denied[:, tick] = movement & switches[:, tick] & (counter == 0) & (delay > 0)
                counter = np.where(switches[:, tick], delay, counter)
                eligible[:, tick] = movement & (counter == 0)
                counter = np.where(initial, 0, np.where(counter == 0, delay, np.maximum(counter - 1, 0)))
            # This clock measures eligibility, not realized movement or causal return.
            round_id = a["round_idx"].astype(int)
            ep, tick = np.nonzero(alive)
            count = np.bincount(ep * 20 + round_id[alive], switches[alive], minlength=len(profile)*20).reshape(-1, 20)
            success = a["success"][endings].reshape(-1, 20)
            lengths = a["round_time"][endings].reshape(-1, 20)
            start = np.zeros((len(profile), 20), int)
            stop = np.nonzero(endings)[1].reshape(-1, 20)
            start[:, 1:] = stop[:, :-1] + 1
            run_lengths, short_runs = [], []
            for episode in range(len(profile)):
                for rnd in range(20):
                    first, last = start[episode, rnd], stop[episode, rnd]
                    changes = np.flatnonzero(switches[episode, first:last+1]) + first
                    bounds = np.r_[first, changes, last+1]
                    dwell = np.diff(bounds)
                    goals = 2-a["partner_assignment"][episode, bounds[:-1]]
                    delays = pool[profile[episode], goals]
                    # Censor the last segment; initial segment begins with ctr=0.
                    if len(dwell) > 2:
                        run_lengths.extend(dwell[1:-1])
                        short_runs.extend(dwell[1:-1] <= delays[1:-1])
            for orientation, selected in (("all", np.ones(len(profile), bool)),
                ("partner_red_faster", pool[profile, 0] < pool[profile, 1]),
                ("partner_blue_faster", pool[profile, 1] < pool[profile, 0])):
                n = int(selected.sum())
                if not n:
                    continue
                denominator = (alive & ~a["is_t0"])[selected].sum()
                ret = np.where(alive, a["rewards"], 0)[selected].sum(axis=1).mean()
                rows.append(dict(version=version, condition=condition, seed=seed, population=population,
                    orientation=orientation, n_episodes=n, episode_return=float(ret),
                    success=float(success[selected].mean()), switches_per_round=float(count[selected].mean()),
                    mean_round_steps=float(lengths[selected].mean()),
                    switch_rate_per_movement_tick=float(switches[selected].sum()/denominator),
                    eligible_partner_move_fraction=float(eligible[selected].sum()/denominator),
                    switch_denied_ready_move_fraction=float(denied[selected].sum()/denominator),
                    switches_successful_rounds=float(count[selected][success[selected]].mean()) if success[selected].any() else None,
                    switches_failed_rounds=float(count[selected][~success[selected]].mean()) if (~success[selected]).any() else None,
                    middle_assignment_segments=len(run_lengths) if orientation=="all" else None,
                    median_middle_assignment_dwell=float(np.median(run_lengths)) if orientation=="all" and run_lengths else None,
                    middle_segments_ended_before_first_eligible_move=float(np.mean(short_runs)) if orientation=="all" and short_runs else None))
            if version == "v1_matched" or "noinfluence" in condition:
                if switches.any():
                    raise ValueError("Fixed allocation switched")
        print(f"Audited rollouts: {version} {condition} seed {seed}", flush=True)
    write_csv(out / "rollout_diagnostics.csv", rows)
    return rows


def figures(out, rows, curves):
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(4, 3, figsize=(14, 12), sharex=True, sharey="row")
    for ci, condition in enumerate(CONDITIONS):
        for vi, version in enumerate(VERSIONS):
            ax = axes[ci, vi]
            for seed in range(1, 6):
                data = curves[(version, condition, seed)]
                ax.plot(data["env_steps"][45:]/1e6, trailing(data["episode_return"]),
                        color=colors[seed-1], lw=1.25, label=str(seed))
            ax.axhline(0, color="#999", lw=.6)
            ax.grid(alpha=.2)
            if vi == 0:
                ax.set_ylabel(LABELS[ci] + "\nTraining episode return")
            if ci == 0:
                ax.set_title(version)
            if ci == 3:
                ax.set_xlabel("Environment transitions (millions)")
    fig.suptitle("Weak seeds start behind and recover partially; final performance stays near each seed's best", fontsize=14)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=10, title="Learner seed")
    fig.tight_layout(rect=(0, .065, 1, .97))
    fig.savefig(out / "per_seed_learning.png", dpi=160)
    fig.savefig(out / "per_seed_learning.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for vi, version in enumerate(VERSIONS):
        group = [r for r in rows if r["version"] == version and r["condition"] == "rnn_diverse_influence"]
        for r in group:
            x, y = r["novel_final_partner_faster"], r["novel_episode_return"]
            axes[vi].scatter(x*100, y, color=colors[r["seed"]-1], s=55)
            axes[vi].annotate(str(r["seed"]), (x*100, y), xytext=(4, 3), textcoords="offset points", fontsize=9)
        axes[vi].set_title(version)
        axes[vi].set_xlabel("Partner ends on its faster goal (%)")
        axes[vi].set_xlim(35, 103)
        axes[vi].set_ylim(5, 18)
        axes[vi].axvline(50, color="#999", ls=":")
        axes[vi].grid(alpha=.2)
    axes[0].set_ylabel("Novel-partner episode return")
    fig.suptitle("Diverse RNN + influence: color specialization versus partner-dependent assignment")
    fig.tight_layout()
    fig.savefig(out / "assignment_and_return.png", dpi=160)
    plt.close(fig)


def intervention_evidence(out):
    rows = []
    for version in ("v2", "v3"):
        path = out / f"switch_intervention_{version}_episodes.csv"
        if not path.exists():
            continue
        groups = {}
        for r in read_csv(path):
            if int(r["rounds"]) != 20:
                raise ValueError("Incomplete intervention")
            groups.setdefault((r["condition"], int(r["seed"])), {}).setdefault(
                r["intervention"], {})[(int(r["profile"]), int(r["repetition"]))] = r
        for (condition, seed), variants in groups.items():
            base, held = variants["baseline"], variants["hold_after_first_request"]
            if set(base) != set(held):
                raise ValueError("Unpaired intervention episodes")
            keys = sorted(k for k in base if k[0] != 0)
            b = np.array([float(base[k]["episode_return"]) for k in keys])
            h = np.array([float(held[k]["episode_return"]) for k in keys])
            n_profiles = len({k[0] for k in keys})
            delta = (h-b).reshape(n_profiles, -1)
            rng = np.random.default_rng(20261015)
            draws = rng.integers(0, delta.shape[1], (10000, n_profiles, delta.shape[1]))
            bootstrap = delta[np.arange(n_profiles)[None, :, None], draws].mean(axis=(1, 2))
            lo, hi = np.quantile(bootstrap, (.025, .975))
            rows.append(dict(version=version, condition=condition, seed=seed, population="novel",
                profiles=n_profiles, n_episodes=len(keys), baseline_return=float(b.mean()),
                held_return=float(h.mean()), paired_difference=float(delta.mean()),
                stratified_episode_bootstrap_low=float(lo), stratified_episode_bootstrap_high=float(hi),
                note="Fixed policies; only two episodes per profile. Interval excludes learner-seed uncertainty."))
    if rows:
        write_csv(out/"switch_intervention_comparison.csv", rows)
    return rows


def report(out, rows, provenance, rollout_rows):
    maximum_drop = max(r["best_to_final_drop"] for r in rows)
    counts = {}
    for version in VERSIONS:
        group = [r for r in rows if r["version"] == version and r["condition"]=="rnn_diverse_influence"]
        counts[version] = dict(n=10,
            final_partner_faster_above_80pct=sum(r["novel_final_partner_faster"] >= .8 for r in group),
            initial_choice_above_98pct_one_color=sum(max(r["novel_initial_ego_red"], 1-r["novel_initial_ego_red"]) >= .98 for r in group))
    interventions = intervention_evidence(out)
    result = dict(policies=120, updates=109800, complete=True,
        max_best46_to_final46_return_drop=maximum_drop,
        policies_with_best_to_final_drop_above_1=sum(r["best_to_final_drop"] > 1 for r in rows),
        main_condition_strategy_counts=counts,
        reward_identity_max_abs_error=max(abs(r[p+"_return_identity_error"]) for r in rows for p in ("familiar", "novel")),
        source_files={str(p): sha256(p) for p in (Path(__file__), HISTORY/"updates.csv", HISTORY/"summary.json",
            EVALUATION/"assignment_behaviors_all_seeds/novel/per_seed_summary.csv",
            EVALUATION/"all_5_seeds_step_penalty_0.03/familiar/per_seed.csv", EVALUATION/"all_5_seeds_step_penalty_0.03/novel/per_seed.csv", PROBES)},
        rollout_audited_policies=120 if rollout_rows else 0,
        intervention_policy_count=len(interventions),
        caveats=["Training histories are retrospective; no training ablation or rescue training was run. Separate small final-policy interventions were evaluated.",
            "Training return windows weight PPO updates equally; completed-episode counts unavailable.",
            "Final checkpoints only; evaluation trajectories do not reveal earlier policies' behavior.",
            "Entropy and losses are PPO-epoch means, not exact pre-update behavior-policy values.",
            "No KL, clip fraction, explained variance, gradient norms, phase advantages, or optimizer checkpoints were saved.",
            "PPO minimizes a discounted objective; displayed returns are undiscounted.",
            "The 24-profile familiar evaluation includes 23 profiles absent from single-partner policy training; the intervention also isolates its actual (1,4) training profile.",
            "Faster-goal assignment is a strategy proxy, not an exact oracle for online switching trajectories.",
            "V2/V3 no-influence histories and behaviors are identical; do not count them as independent replication.",
            "Switch-delay replay measures movement eligibility, not realized movement or causal return gains.",
            "Assignment dwell excludes initial and right-censored final segments.",
            "Early/final correlations describe five seeds per cell, are post hoc, and are not causal evidence."],
        references=[dict(title=t,url=u) for t,u in SOURCES], runs=provenance)
    (out/"audit.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    selected = [("v1_matched", "rnn_diverse_influence", 2), ("v1_matched", "rnn_diverse_influence", 7),
                ("v2", "rnn_diverse_influence", 8), ("v3", "rnn_diverse_influence", 4),
                ("v2", "rnn_single_influence", 8), ("v3", "rnn_single_influence", 4),
                ("v2", "rnn_diverse_noinfluence", 4)]
    table = ""
    for key in selected:
        r = next(x for x in rows if identity(x)==key)
        table += (f"<tr><td>{key[0]}</td><td>{key[1]}</td><td>{key[2]}</td>"
            f"<td>{r['return_updates_47_92']:.2f}</td><td>{r['final46_training_return']:.2f}</td>"
            f"<td>{r['novel_episode_return']:.2f}</td><td>{100*r['novel_success']:.1f}%</td>"
            f"<td>{r['novel_switches_per_round']:.2f}</td></tr>")
    references = "".join(f'<li><a href="{u}">{html.escape(t)}</a></li>' for t,u in SOURCES)
    caveats = "".join(f"<li>{html.escape(x)}</li>" for x in result["caveats"])
    intervention_section = ""
    if interventions:
        intervention_table = "".join(f"<tr><td>{r['version']}</td><td>{r['condition']}</td><td>{r['seed']}</td>"
            f"<td>{r['baseline_return']:.2f}</td><td>{r['held_return']:.2f}</td>"
            f"<td>{r['paired_difference']:+.2f}</td></tr>" for r in interventions)
        intervention_section = f"""<h2>A direct test did not support a simple switching-only explanation</h2>
<p>I evaluated eight frozen policies in their exact source environments, pairing two episodes per each of 22 novel profiles
plus the single-training profile (1,4). Each episode retained 20 rounds. The intervention allowed initialization and the first
movement request, then held that assignment for the remaining round. The policy still sampled movement, and all subsequent
observations and GRU states were recomputed from the intervened trajectory. No policy was retrained.</p>
<table><tr><th>Version</th><th>Condition</th><th>Seed</th><th>Novel baseline return</th><th>With assignment held</th><th>Paired change</th></tr>{intervention_table}</table>
<p>Assignment holding did not rescue the weak policies. V2 single-partner seed 8 deteriorated substantially, and the
strong V2 controls also deteriorated. Switching is therefore part of these learned controllers, not an isolated nuisance
that can simply be removed. This rejects this particular rescue hypothesis; it does not establish that every switch is
helpful, or explain why PPO learned these strategies. The small evaluation panel is exploratory, with no new independent
learner-seed replications. Bootstrap intervals in the CSV resample paired episodes within profiles only.</p>
<p><a href="switch_intervention_comparison.csv">Paired intervention comparisons and intervals</a>;
<a href="switch_intervention_v2_provenance.json">V2 provenance</a>;
<a href="switch_intervention_v3_provenance.json">V3 provenance</a>.</p>"""
    document = f"""<!doctype html><meta charset="utf-8"><title>PPO seed diagnosis</title>
<style>body{{font:16px/1.55 system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#202634}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;font-size:14px;width:100%}}td,th{{border:1px solid #ccd3dd;padding:7px;text-align:left}}img{{width:100%}}code{{background:#edf0f5}}li{{margin:7px 0}}</style>
<h1>v1_matched / v2 / v3: what happened to the weak seeds?</h1>
<p><b>Evidence favors early strategy lock-in and incomplete recovery over late catastrophic reward collapse.</b>
All 120 runs completed 915 PPO updates (59,965,440 transitions each). The largest gap between any seed's best
46-update training-return average and its final 46-update average is {maximum_drop:.3f}. No run lost more than one return unit by this measure.</p>
<p>There was one larger temporary regression: V2 diverse RNN seed 8 lost about 3.59 return units from an already poor
earlier running peak, bottoming near update 272 (17.8M steps), then recovered to its best range by the end.
The final-window result therefore rules against a persistent late collapse, not every transient training regression.</p>
<p>Behavior traces here are evaluations of the final trained policies across the 20 rounds of a partner episode.
They are not checkpoints through training. Training trajectories below come separately from the saved PPO logs.</p>
<img src="per_seed_learning.png" alt="All per-seed learning histories">
<h2>Different failure modes require different explanations</h2>
<p><b>Slow learning:</b> v1_matched diverse RNN seed 2 starts far behind, improves for most of training, but never develops capability-sensitive allocation.
V2 diverse RNN seeds 4 and 8, V3 diverse RNN seed 4, and some no-influence seeds show similar early deficits.
Poor no-influence seeds cannot be explained by switching. Weakness is already present during training, so this is not solely novel-partner generalization.</p>
<p><b>Suboptimal specialization:</b> v1_matched diverse RNN seeds 4, 6, and 7 choose one color almost always,
complete nearly 99% of novel rounds, and assign the partner its faster goal about 50% of the time.
{counts['v1_matched']['final_partner_faster_above_80pct']} of ten v1_matched main policies exceed 80% faster-goal assignment,
compared with {counts['v2']['final_partner_faster_above_80pct']} of ten V2 and {counts['v3']['final_partner_faster_above_80pct']} of ten V3.
All five V3 main policies choose one initial color at least 98% of the time. Good navigation and high success can therefore coexist with failure to learn partner-dependent allocation.</p>
<p>Early commitment is not necessarily permanent: v1_matched seeds 3 and 5 were almost exclusively one color around 3–6M steps,
then later developed partner-dependent choices. Conversely, seeds 6 and 7 learned navigation quickly and retained fixed-color strategies.
Early return therefore does not reliably identify which seeds will learn partner adaptation; the per-cell rank correlations are saved separately.</p>
<img src="assignment_and_return.png" alt="Final allocation adaptation and return by learner seed">
<p><b>Switching-heavy policies and transfer:</b> V2 single-partner seed 8 improves its training return to about 9.94,
but earns -3.65 on novel partners with about 27.16 switches per round and 50.05% success.
The environment resets the partner's delay before movement on every actual switch. Frequent switching can repeatedly cancel ready movement opportunities.
This mechanism is present in the code; the optional scalar rollout audit measures its activation. It does not estimate a closed-loop intervention's return.</p>
<table><tr><th>Version</th><th>Condition</th><th>Seed</th><th>Train ~3–6M</th><th>Train final ~3M</th><th>Novel return</th><th>Novel success</th><th>Novel switches/round</th></tr>{table}</table>
{intervention_section}
<h2>Mechanisms supported by the implementation</h2>
<p>Only the ego is learned; the partner is a fixed scripted BFS agent. This is recurrent masked categorical PPO against a fixed partner population,
so simultaneous co-learner nonstationarity is not an explanation. V1_matched stores the pre-observation reset flag and uses it for replay;
V2 and V3 do the same. Historical V1_matched's off-by-one replay reset is already corrected in these experiments.</p>
<p>The per-seed CSV can join the current specialist-only relative-speed probes (red faster versus blue faster, chance 1/2).
These probes hold out episodes within each original profile. Decodability at round 20 does not establish timely
availability before assignment or causal use by the actor.</p>
<p>The actor uses one flat joint movement/allocation softmax. V1_matched allows two choices at initialization and five movement choices afterward;
online versions allow ten joint choices in movement. Entropy is averaged over all ticks, so the initial allocation decision receives
roughly one tick's weight per round. Low initial-choice entropy and high movement entropy can coexist. Entropy decline by itself is not evidence of failure.</p>
<p>Four PPO epochs and 64 minibatches produce 256 optimizer steps per 65,536-transition rollout. Each recurrent minibatch contains only
four full 256-step worker sequences. This is a concrete sensitivity to examine with KL and gradient measurements, not proof of excessive updates.
Cosine learning-rate decay approaches zero by 60M steps. Slow seeds see fewer completed rounds and successful outcomes at a fixed transition budget,
providing a plausible feedback loop for partial recovery. With gamma=.99 and lambda=.95, GAE's direct residual weight decays as .9405 per tick
(half-life about 11.3 ticks); memory learning over many rounds depends on the critic and truncated 256-step replay.</p>
<p>Executed reward is +1 on success and -.01 otherwise, including timeouts. Every final evaluation satisfies
<code>episode return = 1.01 × successes − .01 × episode length</code> within floating-point tolerance.
These data provide no positive evidence of an alternative reward exploit. Identifying a particular optimizer or exploration cause requires new measurements.</p>
<h2>Concrete next experiments, following standard PPO practice</h2>
<ol><li><b>Instrument short reproductions first.</b> Save pre/post-update approximate KL, clip fraction, ratio extremes, critic explained variance,
unclipped actor/critic gradient norms, and per-phase entropy and advantage statistics. Before any update on a rollout, verify old/new action-probability ratios equal one
across episode and rollout boundaries and across minibatch permutations. Preserve optimizer state, RNG, worker states, scheduler cursors and periodic checkpoints.</li>
<li><b>Evaluate fixed panels during training.</b> At regular checkpoints use identical layouts, partner profiles and evaluation RNG streams.
Record training-profile and novel-profile success, length, initial/final allocation, actual switches and delay-clock disruption separately.
Keep seed as the unit of replication; retain weak seeds and the identical V2/V3 no-influence control.</li>
<li><b>Test early commitment.</b> Compare an initial-choice entropy coefficient or entropy target separately from movement entropy, or factor movement and allocation heads.
Use matched learner seeds and schedules. Track whether partner-sensitive allocation emerges, rather than only whether return rises.</li>
<li><b>Test update aggressiveness.</b> In separate paired arms reduce learning rate or PPO epochs, or increase sequences per minibatch.
Add KL-based early stopping only after measuring the actual KL distribution. Do not transplant a universal threshold from another task.</li>
<li><b>Extend the switching interventions carefully.</b> The assignment-holding pilot did not rescue weak policies.
Test minimum dwell or targeted switch suppression on a larger fixed panel, recomputing policy observations and recurrent state at every step.
Compare against an oracle or scripted navigation controller. Replaying fixed movement actions after changing allocation is not a valid policy intervention.</li>
<li><b>Test recovery versus permanent lock-in.</b> From newly saved early optimizer checkpoints, branch weak seeds into baseline continuation,
slower learning-rate decay, and targeted exploration continuation. A longer run with the original cosine schedule already at zero cannot test recovery fairly.
Do not label these on-policy results as established primacy bias; the classic reset evidence primarily concerns off-policy replay methods.</li></ol>
<h2>Scope and limitations</h2><ul>{caveats}</ul>
<p>CSV evidence: <a href="per_seed.csv">per seed</a>, <a href="training_windows.csv">training windows</a>,
<a href="early_final_correlations.csv">within-condition early/final correlations</a>,
<a href="audit.json">provenance and checks</a>.
{"<a href='rollout_diagnostics.csv'>Scalar rollout and delay-clock audit</a>." if rollout_rows else ""}</p>
<h2>Primary methodological references</h2><ul>{references}</ul>"""
    (out/"report.html").write_text(document)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "train/seed_diagnostics/v1_matched_v2_v3")
    parser.add_argument("--rollouts", action="store_true")
    parser.add_argument("--reuse-rollouts", action="store_true",
                        help="Reuse this audit's saved scalar diagnostics after checking input provenance")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows, curves, provenance = audit_histories(out)
    print("Verified all 120 training histories and reward identities", flush=True)
    if args.rollouts and args.reuse_rollouts:
        parser.error("Choose --rollouts or --reuse-rollouts")
    rollout_rows = audit_rollouts(out) if args.rollouts else []
    if args.reuse_rollouts:
        rollout_rows = read_csv(out/"rollout_diagnostics.csv")
        expected = {(v,c,s) for v in VERSIONS for c in CONDITIONS for s in range(1,6)}
        for population in ("train", "test"):
            observed = {identity(r) for r in rollout_rows if r["population"]==population and r["orientation"]=="all"}
            if observed != expected:
                raise ValueError("Incomplete saved scalar audit")
    if rollout_rows:
        metadata = json.loads((EVALUATION/"assignment_behaviors_all_seeds/novel/analysis.json").read_text())
        for record in metadata["sources"]:
            path = resolve_path(record["source"]["path"])
            stat = path.stat()
            if stat.st_size != record["source"]["size"] or stat.st_mtime_ns != record["source"]["mtime_ns"]:
                raise ValueError("Rollout changed since original scalar audit")
            if sha256(resolve_path(record["source"]["summary"])) != record["source"]["summary_sha256"]:
                raise ValueError("Rollout summary changed")
        (out/"rollout_provenance.json").write_text(json.dumps(metadata["sources"], indent=2)+"\n")
    figures(out, rows, curves)
    result = report(out, rows, provenance, rollout_rows)
    print(json.dumps({k: result[k] for k in ("policies", "updates", "max_best46_to_final46_return_drop",
        "main_condition_strategy_counts", "reward_identity_max_abs_error")}, indent=2))


if __name__ == "__main__":
    main()
