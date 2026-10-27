"""Shared representation utilities for relative-speed and sliding-observation analyses.

Contains validated rollout loading, terminal-inclusive GRU features, episode
splits, and the probe utilities shared with sliding-observation experiments.
The obsolete standalone numeric-delay analysis and report entry points were
removed. The relative-speed producer is eval/relative_speed_representation.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd

CONDITIONS = (
    "rnn_diverse_influence", "rnn_single_influence", "rnn_diverse_noinfluence",
)
CONDITION_LABELS = dict(zip(CONDITIONS, (
    "Multi-partner RNN", "Single-partner RNN", "No-influence RNN",
)))
REFERENCE_TIMES = np.array([1, 50, 100, 150, 200, 250, 300, 350, 400])
TARGETS = ("d_R", "d_B")
N_ROUNDS, HIDDEN_DIM, N_EPISODES, N_REPS = 20, 128, 920, 20
PROBE_STEPS, PROBE_LR, BOOTSTRAP_RESAMPLES = 1000, 1e-2, 10_000
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from repo_paths import save_numpy_archive
DEFAULT_TRAINING_SEEDS = tuple(range(1, 6))


def require(condition: bool, message: str) -> None:
    """Assertions that remain active even when Python is run with -O."""
    if not condition:
        raise ValueError(message)


def capability_populations() -> dict[str, set[tuple[int, int]]]:
    # Load the authoritative constants without executing jaxmarl/__init__.py
    # (which imports unrelated environments and their training dependencies).
    path = REPO_ROOT / "jaxmarl/environments/coordination_grid/capability_populations.py"
    spec = importlib.util.spec_from_file_location("grid_capability_populations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {"train": set(module.TRAIN_CAPABILITY_PAIRS),
            "test": set(module.TEST_CAPABILITY_PAIRS)}


def discover_rollout_files(eval_dir: Path, seeds=DEFAULT_TRAINING_SEEDS,
                           seed_directories=None) -> dict[tuple[str, int], dict[str, Path]]:
    require(len(seeds) >= 2 and len(set(seeds)) == len(seeds), "Require distinct policy seeds")
    files = {}
    missing = []
    for condition in CONDITIONS:
        for seed in seeds:
            slices = {}
            for name in ("train", "test"):
                if seed_directories is None:
                    matches = sorted(eval_dir.rglob(f"{condition}_seed{seed}_{name}.h5"))
                else:
                    path = Path(seed_directories[str(seed)]) / f"{condition}_seed{seed}_{name}.h5"
                    matches = [path] if path.is_file() else []
                require(len(matches) <= 1, f"Ambiguous rollouts for {condition}, seed {seed}, {name}: {matches}")
                if not matches:
                    missing.append(f"{condition}_seed{seed}_{name}.h5")
                else:
                    slices[name] = matches[0]
            files[condition, seed] = slices
    if missing:
        raise FileNotFoundError(
            f"Missing RNN rollout files below {eval_dir}:\n  " + "\n  ".join(missing)
            + "\nGenerate ONLY the missing/malformed checkpoint slices with "
            "eval/evaluate_partner_modelling.py --save_hidden "
            "--n_episodes_per_capability 20; existing rollouts must be reused."
        )
    require(len(files) == len(CONDITIONS) * len(seeds), "Incorrect recurrent checkpoint count")
    return files


def episode_valid_length(dones: np.ndarray) -> int:
    # Audit guide:
    # Find the first final done and include that transition in the valid length. A file
    # with no final done is incomplete and rejected. All feature averages must use this
    # valid prefix rather than the whole fixed scan.
    #
    terminal = np.flatnonzero(dones)
    require(len(terminal) > 0, "Episode has no final done; cannot infer a valid trajectory")
    return int(terminal[0]) + 1


def validate_allocation_protocols(files: dict) -> str:
    """Reject combined fixed-allocation and online-allocation experiments."""
    # Audit guide:
    # Read HDF5 protocol attributes and require every input file in an analysis to
    # belong to one allocation design. Historical files without the attribute are
    # classified fixed_allocation. Separate output folders protect interpretation of results
    # from different protocols.
    #
    protocols = set()
    for slices in files.values():
        for path in slices.values():
            with h5py.File(path, "r") as handle:
                protocol = handle.attrs.get("allocation_protocol", "fixed_allocation")
                if isinstance(protocol, bytes):
                    protocol = protocol.decode()
                require(protocol in ("fixed_allocation", "online_v2", "online_v3"),
                        f"{path}: unknown allocation protocol {protocol!r}")
                protocols.add(protocol)
    require(len(protocols) == 1,
            "Mixed allocation protocols: analyze fixed_allocation, online_v2 and online_v3 separately, "
            "using separate --eval-dir and --out-dir paths.")
    return protocols.pop()


def prefix_mean_hidden(hidden: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    # Audit guide:
    # Average the hidden vectors from episode start through each requested time cutoff.
    # Clip the cutoff to the valid episode length; later cutoffs of a completed episode
    # reuse its complete valid history. This is one feature vector per episode/cutoff,
    # not independent observations at every tick.
    #
    require(hidden.ndim == 2 and len(hidden) > 0, "Empty/invalid hidden trajectory")
    cutoffs = np.minimum(np.asarray(cutoffs, dtype=int), len(hidden))
    require(bool(np.all(cutoffs > 0)), "Prefix cutoffs must be positive")
    prefix = np.cumsum(hidden, axis=0, dtype=np.float64)
    return (prefix[cutoffs - 1] / cutoffs[:, None]).astype(np.float32)


def round_prefix_mean_hidden(hidden: np.ndarray, round_idx: np.ndarray,
                             round_done: np.ndarray) -> np.ndarray:
    # Audit guide:
    # Locate all 20 ordered round endings and average the whole episode prefix through
    # each ending. These are cumulative history features, not averages of only the named
    # round. Check the final round endpoint against the first final done.
    #
    require(len(round_idx) == len(hidden) == len(round_done), "Round/hidden lengths differ")
    ends = np.flatnonzero(round_done)
    require(len(ends) == N_ROUNDS, f"Expected 20 valid round ends; found {len(ends)}")
    require(np.array_equal(round_idx[ends], np.arange(N_ROUNDS)),
            "Round ends must identify rounds 0..19 in order")
    require(int(ends[-1]) == len(hidden) - 1, "Final done does not end round 19")
    expected_idx = np.repeat(np.arange(N_ROUNDS), np.diff(np.r_[0, ends + 1]))
    require(np.array_equal(round_idx, expected_idx), "Round indices inconsistent with round_done")
    return prefix_mean_hidden(hidden, ends + 1)


def final50_mean_hidden(hidden: np.ndarray) -> np.ndarray:
    # Audit guide:
    # Average at most the last 50 valid states of a partner episode for UMAP. Short
    # episodes contribute all their states. This final-history view is distinct from
    # prefix-mean probe features.
    #
    require(len(hidden) > 0, "Cannot average an empty episode")
    return hidden[-50:].mean(axis=0, dtype=np.float64).astype(np.float32)


@dataclass
class RolloutDataset:
    labels: np.ndarray
    rollout_rep: np.ndarray
    capability_slice: np.ndarray
    lengths: np.ndarray
    timestep_features: np.ndarray
    round_features: np.ndarray
    final50_features: np.ndarray
    episodes: pd.DataFrame
    validation: dict


def validate_rollout_dataset(data: RolloutDataset) -> None:
    require(data.labels.shape == (N_EPISODES, 2), "Expected 920 episodes with two targets")
    pairs, counts = np.unique(data.labels, axis=0, return_counts=True)
    require(len(pairs) == 46 and np.all(counts == N_REPS),
            "Expected 46 unique profiles, exactly 20 repetitions each")
    require(np.array_equal(np.unique(data.labels), np.arange(10)), "Delay labels must be 0..9")
    for pair in pairs:
        mask = np.all(data.labels == pair, axis=1)
        require(np.array_equal(np.sort(data.rollout_rep[mask]), np.arange(N_REPS)),
                f"Duplicate/missing repetition within profile {pair}")
    for features, steps in ((data.timestep_features, len(REFERENCE_TIMES)),
                            (data.round_features, N_ROUNDS)):
        require(features.shape == (steps, N_EPISODES, HIDDEN_DIM), "Invalid averaged feature shape")
        require(bool(np.isfinite(features).all()), "Non-finite averaged hidden states")
    require(data.final50_features.shape == (N_EPISODES, HIDDEN_DIM), "Invalid final-50 shape")
    require(bool(np.isfinite(data.final50_features).all()), "Non-finite final-50 hidden states")


def load_checkpoint_rollouts(paths: dict[str, Path]) -> RolloutDataset:
    """Read in small batches, check full scans, retain only valid prefix means."""
    # Audit guide:
    # Validate episode counts, capability constancy, terminal flags, hidden dimensions,
    # and round boundaries while reading HDF5 in small batches. Retain feature means
    # rather than every full trajectory. Order episodes by capability and repetition so
    # one shared split applies consistently across policy seeds.
    #
    population = capability_populations()
    labels, reps, slices, lengths, time_features, round_features, final_features = [], [], [], [], [], [], []
    rows, sources = [], []
    for name in ("train", "test"):
        path = paths[name]
        try:
            with h5py.File(path, "r") as handle:
                required = {"hidden_state", "capability", "dones", "round_idx", "round_done"}
                require(required <= set(handle), f"{path}: missing datasets {required - set(handle)}")
                E, T = handle["dones"].shape
                require(E == len(population[name]) * N_REPS, f"{path}: invalid episode count {E}")
                require(handle["hidden_state"].shape == (E, T, HIDDEN_DIM), f"{path}: expected hidden dimension 128")
                require(handle["capability"].shape == (E, T, 2), f"{path}: invalid capability shape")
                for field in ("round_idx", "round_done"):
                    require(handle[field].shape == (E, T), f"{path}: invalid {field} shape")
                indices = handle["capability_index_per_ep"][:] if "capability_index_per_ep" in handle else None
                pool = handle["capability_pool"][:] if "capability_pool" in handle else None
                require((indices is None) == (pool is None), f"{path}: incomplete capability ordering metadata")
                if indices is not None:
                    require(indices.shape == (E,) and pool.shape == (len(population[name]), 2),
                            f"{path}: invalid capability ordering metadata shape")
                    require(bool(np.all((indices >= 0) & (indices < len(pool)))), f"{path}: capability index out of range")
                    require({tuple(p) for p in pool} == population[name], f"{path}: incorrect capability_pool")
                counts = {}
                for start in range(0, E, 16):
                    stop = min(start + 16, E)
                    block = {k: handle[k][start:stop] for k in required}
                    require(bool(np.isfinite(block["hidden_state"]).all()),
                            f"{path}: NaN/Inf in hidden-state scan at episodes {start}:{stop}")
                    for local, e in enumerate(range(start, stop)):
                        L = episode_valid_length(block["dones"][local])
                        cap = block["capability"][local, 0]
                        require(bool(np.all(block["capability"][local, :L] == cap)),
                                f"{path}, episode {e}: capability changes before terminal")
                        require(bool(np.all(cap == cap.astype(int))), f"{path}: non-integer cooldown")
                        pair = tuple(int(x) for x in cap)
                        require(pair in population[name], f"{path}: unexpected {name} profile {pair}")
                        if indices is not None:
                            require(np.array_equal(pool[indices[e]], cap), f"{path}: capability_index_per_ep disagrees with labels")
                        rep = counts.get(pair, 0)
                        counts[pair] = rep + 1
                        # The terminal step is included; scan padding after it never enters features.
                        hidden = block["hidden_state"][local, :L]
                        tf = prefix_mean_hidden(hidden, REFERENCE_TIMES)
                        rf = round_prefix_mean_hidden(hidden, block["round_idx"][local, :L],
                                                      block["round_done"][local, :L])
                        ff = final50_mean_hidden(hidden)
                        labels.append(pair); reps.append(rep); slices.append(name); lengths.append(L)
                        time_features.append(tf); round_features.append(rf); final_features.append(ff)
                        rows.append({"source_file": str(path.resolve()), "source_episode": e,
                                     "capability_slice": name, "capability_index": int(indices[e]) if indices is not None else -1,
                                     "d_R": pair[0], "d_B": pair[1], "rollout_rep": rep,
                                     "valid_length": L, "scan_length": T})
                require(set(counts) == population[name] and set(counts.values()) == {N_REPS},
                        f"{path}: expected exactly 20 rollouts for every authoritative profile")
                stat = path.stat()
                sources.append({"path": str(path.resolve()), "size_bytes": stat.st_size,
                                "mtime_ns": stat.st_mtime_ns, "episodes": E, "scan_steps": T,
                                "rep_source": "episode order within stored capability_index_per_ep" if indices is not None
                                else "episode order within capability pair"})
        except (OSError, ValueError) as error:
            raise ValueError(f"Malformed rollout {path}: {error}. Regenerate this slice only using --save_hidden.") from error
    # Canonical ordering makes random vectors, labels and probe splits paired
    # even if files order their capability profiles differently.
    labels = np.asarray(labels, dtype=np.int64)
    reps = np.asarray(reps, dtype=np.int64)
    order = np.lexsort((reps, labels[:, 1], labels[:, 0]))
    dataset = RolloutDataset(labels[order], reps[order], np.asarray(slices)[order],
                             np.asarray(lengths)[order], np.asarray(time_features)[order].transpose(1, 0, 2),
                             np.asarray(round_features)[order].transpose(1, 0, 2),
                             np.asarray(final_features)[order], pd.DataFrame(rows).iloc[order].reset_index(drop=True),
                             {"sources": sources, "checks_passed": [
                                 "train_and_test_present", "920_episodes", "46_authoritative_profiles",
                                 "20_repetitions_per_profile", "hidden_dimension_128", "capability_constant_on_valid_steps",
                                 "full_hidden_scans_finite", "final_done_every_episode", "terminal_step_included_padding_excluded",
                                 "20_ordered_round_ends_every_episode"],
                              "length_min": int(min(lengths)), "length_max": int(max(lengths)),
                              "length_mean": float(np.mean(lengths)), "episodes_shorter_than_50": int(np.sum(np.asarray(lengths) < 50))})
    validate_rollout_dataset(dataset)
    return dataset


def make_probe_split(analysis_seed: int) -> dict:
    # Audit guide:
    # Permute the 20 repetition indices once and reuse 16 for fitting and four for
    # testing within every profile. This holds out episodes, not capability profiles.
    # Repetition indices describe source order rather than twenty globally shared
    # environment seeds.
    #
    permutation = np.random.default_rng(analysis_seed).permutation(N_REPS)
    return {"analysis_seed": analysis_seed, "rng": "numpy.default_rng (PCG64)",
            "permutation": permutation.tolist(), "train_reps": permutation[:16].tolist(),
            "test_reps": permutation[16:].tolist(),
            "rep_definition": "zero-based occurrence in source episode order within each capability profile",
            "n_train_per_profile": 16, "n_test_per_profile": 4}


def split_masks(data: RolloutDataset, split: dict) -> tuple[np.ndarray, np.ndarray]:
    # Audit guide:
    # Translate the saved repetition split into episode masks and verify 16/4 coverage
    # for every profile. A novel policy-test profile is still represented in probe
    # training here; this analysis does not claim novel-profile transfer by the probe.
    #
    train = np.isin(data.rollout_rep, split["train_reps"])
    test = np.isin(data.rollout_rep, split["test_reps"])
    require(not np.any(train & test) and bool(np.all(train | test)), "Probe split overlaps or omits rollouts")
    require(int(train.sum()) == 736 and int(test.sum()) == 184, "Expected 736 probe train / 184 test episodes")
    for pair in np.unique(data.labels, axis=0):
        mask = np.all(data.labels == pair, axis=1)
        require(int((train & mask).sum()) == 16 and int((test & mask).sum()) == 4,
                f"Profile {pair}: split must be exactly 16/4")
    return train, test


def train_linear_probe(features: np.ndarray, labels: np.ndarray, train: np.ndarray,
                       initialization_seeds: list[int], device: str = "cpu") -> tuple[np.ndarray, dict]:
    """Fit P independent affine 128->10 layers with full-batch Adam.

    Each loss is averaged over its own 736 examples, then summed across
    models. Parameters and Adam moments are independent along the P axis.
    No standardization, regularization, hidden layers, minibatch randomness,
    validation-based tuning, or early stopping is applied.
    """
    # Audit guide:
    # Fit separate affine hidden-state-to-delay classifiers with independent weights and
    # optimizer moments, even though computation is batched. Train for a fixed 1000 Adam
    # steps on the training mask. Use no test-based checkpoint selection, warm starts
    # across cutoffs, scaling, or regularization in this written-protocol
    # implementation.
    #
    import torch
    require(features.ndim == 3 and features.shape[2] == HIDDEN_DIM, "Probe inputs must be averaged 128-D states only")
    P, N, H = features.shape
    require(labels.shape == (P, N) and len(initialization_seeds) == P, "Probe target/seed shapes differ")
    require(bool(np.isfinite(features).all()), "Non-finite probe features")
    require(bool(np.all((labels >= 0) & (labels < 10))), "Probe labels outside 0..9")
    X = torch.as_tensor(np.ascontiguousarray(features), dtype=torch.float32, device=device)
    Y = torch.as_tensor(labels, dtype=torch.long, device=device)
    weights, biases = [], []
    for seed in initialization_seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        bound = H ** -0.5
        weights.append(torch.empty(H, 10).uniform_(-bound, bound, generator=generator))
        biases.append(torch.empty(10).uniform_(-bound, bound, generator=generator))
    W = torch.nn.Parameter(torch.stack(weights).to(device))
    b = torch.nn.Parameter(torch.stack(biases).to(device))
    optimizer = torch.optim.Adam([W, b], lr=PROBE_LR, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=0, foreach=False)
    Xtrain, Ytrain = X[:, train], Y[:, train]
    for _ in range(PROBE_STEPS):
        optimizer.zero_grad(set_to_none=True)
        logits = torch.bmm(Xtrain, W) + b[:, None, :]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 10), Ytrain.reshape(-1),
                                                 reduction="sum") / int(train.sum())
        require(bool(torch.isfinite(loss)), "Linear-probe loss diverged")
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        predictions = (torch.bmm(X, W) + b[:, None, :]).argmax(dim=-1).cpu().numpy()
        losses = torch.nn.functional.cross_entropy(
            (torch.bmm(Xtrain, W) + b[:, None, :]).reshape(-1, 10), Ytrain.reshape(-1),
            reduction="none").reshape(P, -1).mean(dim=1).cpu().numpy()
    return predictions, {"weight": W.detach().cpu().numpy(), "bias": b.detach().cpu().numpy(),
                         "final_train_loss": losses}


def distance_aware_accuracy(truth: np.ndarray, predictions: np.ndarray) -> float:
    # Audit guide:
    # Score a predicted delay by 1 minus its absolute error divided by nine. A near miss
    # gets partial credit, so a high score is not a high exact ten-class accuracy. Lower
    # delay labels represent faster partners.
    #
    return float(np.mean(1.0 - np.abs(predictions - truth) / 9.0))


def evaluate_probe(truth: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> dict:
    require(bool(np.any(mask)), "Cannot evaluate an empty probe subset")
    truth, predictions = truth[mask], predictions[mask]
    return {"n_test": int(mask.sum()), "distance_accuracy": distance_aware_accuracy(truth, predictions),
            "exact_accuracy": float(np.mean(truth == predictions)),
            "mae": float(np.mean(np.abs(predictions - truth)))}


def run_probes(data: RolloutDataset, split: dict, condition: str, seed: int,
               analysis_seed: int, device: str, out_dir: Path) -> tuple[list, list, list]:
    # Audit guide:
    # Fit independent red and blue probes for every timestep/round feature set and save
    # models and scores. Test the same fitted probe on all, familiar, and novel episode
    # subsets. Also shuffle paired delay labels at the last timestep as a diagnostic
    # that breaks their association with real hidden states.
    #
    train, test = split_masks(data, split)
    specs, features, labels, init_seeds = [], [], [], []
    for axis, values, feature_set in (("reference_t", REFERENCE_TIMES, data.timestep_features),
                                      ("round_idx", range(N_ROUNDS), data.round_features)):
        for step, value in enumerate(values):
            for target_idx, target in enumerate(TARGETS):
                specs.append((axis, int(value), target))
                features.append(feature_set[step]); labels.append(data.labels[:, target_idx])
                init_seeds.append(analysis_seed + 10_000 + target_idx)
    # One paired, global label permutation preserves the d_R/d_B joint population,
    # but breaks its association with the hidden trajectory. No grouping fields
    # enter any probe. Compare this final-t=400 diagnostic with Normal baseline.
    shuffled = data.labels[np.random.default_rng(analysis_seed + 20_000).permutation(N_EPISODES)]
    for target_idx, target in enumerate(TARGETS):
        specs.append(("shuffled", 400, target)); features.append(data.timestep_features[-1])
        labels.append(shuffled[:, target_idx]); init_seeds.append(analysis_seed + 10_000 + target_idx)
    predictions, params = train_linear_probe(np.asarray(features), np.asarray(labels), train, init_seeds, device)
    save_numpy_archive(out_dir / "probe_models" / f"{condition}_seed{seed}.npz", compressed=True, **params,
                        axis=np.array([x[0] for x in specs]), step=np.array([x[1] for x in specs]),
                        target=np.array([x[2] for x in specs]), initialization_seed=init_seeds)
    timestep_rows, round_rows, shuffle_rows = [], [], []
    for p, (axis, value, target) in enumerate(specs):
        for subset in ("all", "familiar", "novel"):
            mask = test.copy()
            if subset != "all":
                mask &= data.capability_slice == ("train" if subset == "familiar" else "test")
            row = {"condition": condition, "training_seed": seed, "target": target,
                   "capability_subset": subset, "n_train": int(train.sum()),
                   "probe_initialization_seed": init_seeds[p], "final_train_loss": float(params["final_train_loss"][p]),
                   **evaluate_probe(labels[p], predictions[p], mask)}
            if axis == "reference_t":
                row[axis] = value
                subset_mask = np.ones(N_EPISODES, dtype=bool) if subset == "all" else (
                    data.capability_slice == ("train" if subset == "familiar" else "test"))
                row["fraction_episodes_complete"] = float(np.mean(data.lengths[subset_mask] <= value))
                timestep_rows.append(row)
            elif axis == "round_idx":
                row[axis] = value; round_rows.append(row)
            else:
                row["reference_t"] = value; row["label_shuffle_seed"] = analysis_seed + 20_000
                shuffle_rows.append(row)
    return timestep_rows, round_rows, shuffle_rows


def run_random_baseline(data: RolloutDataset, split: dict, device: str,
                        analysis_seed: int, out_dir: Path) -> pd.DataFrame:
    # Audit guide:
    # Replace hidden states with independent Normal features while retaining true
    # capability labels and the shared episode split. Fit with identical probe settings.
    # This baseline controls for label frequencies and forgiving distance-aware scoring
    # rather than testing a recurrent policy.
    #
    train, test = split_masks(data, split)
    features, labels, seeds, specs = [], [], [], []
    for random_seed in range(5):
        for k, target in enumerate(TARGETS):
            # Independent Normal representation for each target; seed=0 saved
            # explicitly for both targets as the paper's single-realization view.
            features.append(np.random.default_rng(np.random.SeedSequence([random_seed, k])).normal(
                size=(N_EPISODES, HIDDEN_DIM)).astype(np.float32))
            labels.append(data.labels[:, k]); seeds.append(analysis_seed + 10_000 + k)
            specs.append((random_seed, target))
    predictions, params = train_linear_probe(np.asarray(features), np.asarray(labels), train, seeds, device)
    save_numpy_archive(out_dir / "probe_models/random_baseline.npz", compressed=True, **params,
                        random_seed=np.array([x[0] for x in specs]), target=np.array([x[1] for x in specs]))
    rows = []
    for p, (random_seed, target) in enumerate(specs):
        for subset in ("all", "familiar", "novel"):
            mask = test.copy()
            if subset != "all":
                mask &= data.capability_slice == ("train" if subset == "familiar" else "test")
            rows.append({"target": target, "random_vector_seed": random_seed, "capability_subset": subset,
                         "n_train": int(train.sum()), "probe_initialization_seed": seeds[p],
                         "final_train_loss": float(params["final_train_loss"][p]),
                         **evaluate_probe(labels[p], predictions[p], mask)})
    return pd.DataFrame(rows)


def summarize_probes(table: pd.DataFrame, axis: str, bootstrap_seed: int,
                     seeds=DEFAULT_TRAINING_SEEDS) -> pd.DataFrame:
    # The same resampled policy indices are used at every time, task and condition.
    # Audit guide:
    # Aggregate fitted-probe scores across learner seeds with means, sample standard
    # deviations, and bootstrap intervals. The resampling unit is the trained policy
    # seed; repeated rollout examples are not independent trained models.
    #
    indices = np.random.default_rng(bootstrap_seed).integers(0, len(seeds), (BOOTSTRAP_RESAMPLES, len(seeds)))
    rows = []
    for keys, group in table.groupby(["condition", "target", axis, "capability_subset"], sort=True):
        group = group.sort_values("training_seed")
        require(group.training_seed.tolist() == list(seeds),
                "Summary must contain all five independent model seeds" if len(seeds) == 5
                else f"Summary must contain all {len(seeds)} requested independent model seeds")
        row = dict(zip(("condition", "target", axis, "capability_subset"), keys))
        row["n_training_seeds"] = len(seeds)
        for metric in ("distance_accuracy", "exact_accuracy", "mae"):
            values = group[metric].to_numpy()
            low, high = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
            row.update({f"{metric}_mean": float(values.mean()), f"{metric}_std": float(values.std(ddof=1)),
                        f"{metric}_ci_low": float(low), f"{metric}_ci_high": float(high)})
        if "fraction_episodes_complete" in group:
            row["fraction_episodes_complete_mean"] = float(group.fraction_episodes_complete.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_condition_differences(table: pd.DataFrame, axis: str, bootstrap_seed: int,
                                    seeds=DEFAULT_TRAINING_SEEDS) -> pd.DataFrame:
    """Paired descriptive contrasts against the diverse/influence condition."""
    # Audit guide:
    # Compare each control with the diverse influence-enabled RNN using matched nominal
    # learner-seed IDs. Save descriptive paired contrasts and uncertainty. Pairing seed
    # IDs is useful bookkeeping, not proof that every policy received identical
    # experiences.
    #
    indices = np.random.default_rng(bootstrap_seed).integers(0, len(seeds), (BOOTSTRAP_RESAMPLES, len(seeds)))
    rows = []
    for (target, step, subset), group in table.groupby(["target", axis, "capability_subset"]):
        pivot = group.pivot(index="training_seed", columns="condition", values="distance_accuracy").sort_index()
        require(pivot.index.tolist() == list(seeds) and not pivot.isna().any().any(),
                f"Condition contrasts require all {len(seeds)} seeds in every condition")
        for control in CONDITIONS[1:]:
            differences = (pivot[CONDITIONS[0]] - pivot[control]).to_numpy()
            low, high = np.quantile(differences[indices].mean(axis=1), [0.025, 0.975])
            rows.append({"axis": axis, "step": int(step), "target": target, "capability_subset": subset,
                         "condition": CONDITIONS[0], "comparison_condition": control,
                         "distance_accuracy_difference_mean": float(differences.mean()),
                         "difference_ci_low": float(low), "difference_ci_high": float(high),
                         "n_positive_seed_differences": int(np.sum(differences > 0)), "n_training_seeds": len(seeds)})
    return pd.DataFrame(rows)
