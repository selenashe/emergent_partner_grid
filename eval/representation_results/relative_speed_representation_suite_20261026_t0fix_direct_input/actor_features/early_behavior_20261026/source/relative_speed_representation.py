"""Independent binary probes of red-faster versus blue-faster specialist partners.

Representation averaging is imported from the frozen delay-decoding analysis.
The target is categorical; no numeric-delay regression or distance metric is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import representation_analysis as original

CLASS_NAMES = ("Faster red", "Faster blue")
CLASS_COLORS = ("#d94b41", "#3779bf")
CONDITION_COLORS = ("#9380ff", "#e67e00", "#dbb528")
VERSIONS = ("v1_matched", "v2", "v3")
CONDITIONS = original.CONDITIONS


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def relative_classes(pairs):
    """Original specialist profiles only: red faster=0, blue faster=1."""
    pairs = np.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("Expect pairs of red/blue delays")
    if not np.isfinite(pairs).all() or not np.all(pairs == pairs.astype(int)) or np.any((pairs < 0) | (pairs > 9)):
        raise ValueError("Delays must be integers in 0..9")
    if np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError("Equal-delay partners are excluded from this analysis")
    return (pairs[:, 1] < pairs[:, 0]).astype(np.int64)


def masks(repetitions, pairs, split):
    train = np.isin(repetitions, split["train_reps"])
    test = np.isin(repetitions, split["test_reps"])
    if np.any(train & test) or not np.all(train | test):
        raise ValueError("Episode split overlaps or omits episodes")
    for pair in np.unique(pairs, axis=0):
        selected = np.all(pairs == pair, axis=1)
        if (np.sum(selected & train), np.sum(selected & test)) != (16, 4):
            raise ValueError(f"Incorrect profile split: {pair}")
    return train, test


def class_metrics(truth, prediction, require_all=True):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if len(truth) == 0 or truth.shape != prediction.shape:
        raise ValueError("Missing or mismatched predictions")
    if np.any((truth < 0) | (truth >= 2) | (prediction < 0) | (prediction >= 2)):
        raise ValueError("Invalid categorical class")
    confusion = np.bincount(truth * 2 + prediction, minlength=4).reshape(2, 2)
    support = confusion.sum(axis=1)
    if require_all and np.any(support == 0):
        raise ValueError("A two-class metric requires examples of all both classes")
    recalls = np.divide(np.diag(confusion), support, out=np.full(2, np.nan), where=support > 0)
    return dict(accuracy=float((truth == prediction).mean()), balanced_accuracy=float(np.nanmean(recalls)),
                **{f"recall_{k}": float(recalls[k]) for k in range(2)},
                **{f"support_{k}": int(support[k]) for k in range(2)}), confusion


def train_probes(features, labels, train, seeds, steps=1000):
    """Independent affine H→2 softmax heads, full-batch class-balanced Adam."""
    import torch

    features, labels = np.asarray(features), np.asarray(labels)
    P, N, H = features.shape
    if H < 1 or labels.shape != (P, N) or len(seeds) != P or not np.isfinite(features).all():
        raise ValueError("Invalid independent probe inputs")
    if np.any((labels < 0) | (labels >= 2)) or not np.any(train):
        raise ValueError("Invalid two-class labels or empty training set")
    counts = np.stack([np.bincount(y[train], minlength=2) for y in labels])
    if np.any(counts == 0):
        raise ValueError("Cannot train a two-class probe without every class")
    # Weights derived exclusively from training labels. Each class contributes 1/2.
    class_weights = train.sum() / (2 * counts)
    X = torch.as_tensor(np.ascontiguousarray(features), dtype=torch.float32)
    Y = torch.as_tensor(labels, dtype=torch.long)
    weights, biases = [], []
    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        bound = H ** -0.5
        weights.append(torch.empty(H, 2).uniform_(-bound, bound, generator=generator))
        biases.append(torch.empty(2).uniform_(-bound, bound, generator=generator))
    W = torch.nn.Parameter(torch.stack(weights))
    b = torch.nn.Parameter(torch.stack(biases))
    optimizer = torch.optim.Adam([W, b], lr=0.01, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=0, foreach=False)
    Xt, Yt = X[:, train], Y[:, train]
    cw = torch.as_tensor(class_weights, dtype=torch.float32)
    observation_weights = cw.gather(1, Yt)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = torch.bmm(Xt, W) + b[:, None, :]
        losses = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), Yt.reshape(-1),
                                                   reduction="none").reshape(P, -1)
        loss = (losses * observation_weights).sum() / train.sum()
        if not torch.isfinite(loss):
            raise ValueError("Probe loss diverged")
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        probabilities = (torch.bmm(X, W) + b[:, None, :]).softmax(dim=-1).numpy()
        final_logits = torch.bmm(Xt, W) + b[:, None, :]
        final_losses = torch.nn.functional.cross_entropy(final_logits.reshape(-1, 2), Yt.reshape(-1),
            reduction="none").reshape(P, -1)
    return probabilities.argmax(axis=-1), probabilities, dict(weight=W.detach().numpy(), bias=b.detach().numpy(),
        class_weights=class_weights, final_train_loss=(final_losses * observation_weights).mean(dim=1).numpy())


def load_data(record):
    baseline = original.load_checkpoint_rollouts({k: Path(v) for k, v in record["original_rollouts"].items()})
    classes = relative_classes(baseline.labels)
    if not np.array_equal(np.bincount(classes, minlength=2), [460, 460]):
        raise ValueError("Expected 460 red-faster and 460 blue-faster specialist episodes")
    return dict(pairs=baseline.labels, repetitions=baseline.rollout_rep,
        slices=baseline.capability_slice, timestep=baseline.timestep_features,
        round=baseline.round_features, final50=baseline.final50_features,
        episodes=baseline.episodes.copy(), sources=baseline.validation["sources"], classes=classes)


def analyze(manifest_path, index):
    import torch
    torch.set_num_threads(1)
    manifest = json.loads(Path(manifest_path).read_text())
    record = manifest["policies"][index]
    out = Path(record["analysis_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "completion.json").exists():
        done = json.loads((out / "completion.json").read_text())
        if (done["status"] != "completed" or done["episodes"] != 920
                or (done["version"], done["condition"], done["seed"]) != (record["version"], record["condition"], record["seed"])):
            raise ValueError("Existing analysis belongs to different inputs")
        print(f"[reuse analysis] {index} {out}", flush=True)
        return
    print(f"[load] {index} {record['version']} {record['condition']} seed{record['seed']}", flush=True)
    data = load_data(record)
    split = original.make_probe_split(0)
    train, test = masks(data["repetitions"], data["pairs"], split)
    if (train.sum(), test.sum()) != (736, 184):
        raise ValueError("Wrong combined episode counts")
    features = np.concatenate([data["timestep"], data["round"], data["timestep"][-1:]])
    labels = np.broadcast_to(data["classes"], (30, 920)).copy()
    labels[-1] = np.random.default_rng(20000).permutation(labels[-1])
    predictions, probabilities, models = train_probes(features, labels, train, [10000] * 29 + [20000])
    np.savez_compressed(out / "probe_models.npz", **models, predictions=predictions,
        probabilities=probabilities, truth=labels, train=train, test=test, pairs=data["pairs"],
        repetitions=data["repetitions"], time_cutoffs=original.REFERENCE_TIMES,
        round_cutoffs=np.arange(1, 21))
    rows, confusions = [], []
    specs = [("timestep", int(t), "primary") for t in original.REFERENCE_TIMES]
    specs += [("round", r, "primary") for r in range(1, 21)] + [("timestep", 400, "shuffled_labels")]
    for i, (axis, cutoff, kind) in enumerate(specs):
        for subset in ("all", "train", "test"):
            selected = test & (True if subset == "all" else data["slices"] == subset)
            metrics, confusion = class_metrics(labels[i, selected], predictions[i, selected], require_all=subset == "all")
            common = dict(version=record["version"], condition=record["condition"], training_seed=record["seed"],
                axis=axis, cutoff=cutoff, kind=kind, population=subset, n_episodes=int(selected.sum()))
            rows.append(dict(**common, **metrics))
            for truth in range(2):
                for predicted in range(2):
                    confusions.append(dict(**common, true_class=truth, predicted_class=predicted,
                        count=int(confusion[truth, predicted])))
    pd.DataFrame(rows).to_csv(out / "probe_per_seed.csv", index=False)
    pd.DataFrame(confusions).to_csv(out / "confusion_counts.csv", index=False)
    random_features = np.stack([np.random.default_rng(s).standard_normal((920, 128)).astype(np.float32) for s in range(5)])
    random_labels = np.broadcast_to(data["classes"], (5, 920)).copy()
    random_predictions, random_prob, random_models = train_probes(random_features, random_labels, train, [10000] * 5)
    np.savez_compressed(out / "random_models.npz", **random_models, predictions=random_predictions,
        probabilities=random_prob, feature_seeds=np.arange(5))
    control = []
    for s in range(5):
        metrics, _ = class_metrics(data["classes"][test], random_predictions[s, test])
        control.append(dict(version=record["version"], condition=record["condition"], training_seed=record["seed"],
                            random_feature_seed=s, **metrics))
    pd.DataFrame(control).to_csv(out / "random_baseline.csv", index=False)
    data["episodes"]["class"] = data["classes"]
    data["episodes"]["probe_partition"] = np.where(train, "train", "test")
    data["episodes"].to_csv(out / "episode_manifest.csv", index=False)
    write(out / "probe_split.json", split)
    print(f"[umap] {index}", flush=True)
    from umap import UMAP
    embedding = UMAP(n_components=2, n_neighbors=919, min_dist=1.0, metric="euclidean",
                     random_state=42, init="spectral", n_jobs=1).fit_transform(data["final50"])
    table = data["episodes"].copy()
    table["umap_x"], table["umap_y"] = embedding.T
    table.to_csv(out / "umap.csv", index=False)
    plot_umap(table, out, f"{record['version']} · {original.CONDITION_LABELS[record['condition']]} · seed {record['seed']}")
    write(out / "completion.json", dict(status="completed", version=record["version"], condition=record["condition"],
        seed=record["seed"], completed_utc=datetime.now(timezone.utc).isoformat(), episodes=920,
        class_counts=[460, 460], probe_train=736, probe_test=184, n_primary_probes=29,
        sources=data["sources"], umap_parameters=dict(n_neighbors=919, min_dist=1.0, metric="euclidean", random_state=42)))
    print(f"[done] {index} {out}", flush=True)


def plot_umap(table, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    draw_umap(ax, table)
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(out / f"umap_relative_speed.{extension}", dpi=180)
    plt.close(fig)


def draw_umap(ax, table):
    for k, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        selected = table[table["class"] == k]
        ax.scatter(selected.umap_x, selected.umap_y, c=color, label=name, s=8, alpha=0.65, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    for spine in ax.spines.values():
        spine.set_visible(False)


def summary(table):
    rows = []
    rng = np.random.default_rng(30000)
    keys = ["version", "condition", "axis", "cutoff", "kind", "population"]
    for group_keys, group in table.groupby(keys, sort=True):
        if len(group) != 5 or set(group.training_seed) != set(range(1, 6)):
            raise ValueError("Every summary needs five independent policy seeds")
        row = dict(zip(keys, group_keys))
        for metric in ("balanced_accuracy", "accuracy", "recall_0", "recall_1"):
            values = group.sort_values("training_seed")[metric].to_numpy()
            if np.isnan(values).all():
                continue
            if not np.isfinite(values).all():
                raise ValueError("Partial missing seed metrics")
            boot = values[rng.integers(0, 5, size=(10000, 5))].mean(axis=1)
            row.update({f"{metric}_mean": float(values.mean()), f"{metric}_sd": float(values.std(ddof=1)),
                f"{metric}_ci_low": float(np.quantile(boot, 0.025)), f"{metric}_ci_high": float(np.quantile(boot, 0.975))})
        rows.append(row)
    return pd.DataFrame(rows)


def plots(table, summed, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    for axis, xlabel in (("round", "Cumulative rounds"), ("timestep", "Cumulative timesteps")):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
        for ax, version in zip(axs, VERSIONS):
            for condition, color in zip(CONDITIONS, CONDITION_COLORS):
                g = summed[(summed.version == version) & (summed.condition == condition) & (summed.axis == axis)
                           & (summed.kind == "primary") & (summed.population == "all")].sort_values("cutoff")
                ax.plot(g.cutoff, g.balanced_accuracy_mean, c=color, lw=2, label=original.CONDITION_LABELS[condition])
                ax.fill_between(g.cutoff, g.balanced_accuracy_ci_low, g.balanced_accuracy_ci_high, color=color, alpha=0.18)
            ax.axhline(1/2, color="gray", ls="--", lw=1, label="Chance (1/2)")
            ax.set_title(version); ax.set_xlabel(xlabel); ax.set_ylim(0, 1)
            ax.spines[["top", "right"]].set_visible(False)
        axs[0].set_ylabel("Binary balanced accuracy")
        axs[-1].legend(frameon=False, fontsize=8, loc="lower right")
        fig.suptitle("Mean across 5 training seeds; shading: 95% bootstrap CI across seeds", fontsize=11)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(root / f"relative_speed_{axis}_all_seeds.{ext}", dpi=180)
        plt.close(fig)
        with PdfPages(root / f"relative_speed_{axis}_individual_seeds.pdf") as pdf:
            for seed in range(1, 6):
                fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
                for ax, version in zip(axs, VERSIONS):
                    for condition, color in zip(CONDITIONS, CONDITION_COLORS):
                        g = table[(table.version == version) & (table.condition == condition) & (table.training_seed == seed)
                            & (table.axis == axis) & (table.kind == "primary") & (table.population == "all")].sort_values("cutoff")
                        ax.plot(g.cutoff, g.balanced_accuracy, c=color, lw=2, label=original.CONDITION_LABELS[condition])
                    ax.axhline(1/2, c="gray", ls="--", lw=1)
                    ax.set_title(version); ax.set_xlabel(xlabel); ax.set_ylim(0, 1)
                    ax.spines[["top", "right"]].set_visible(False)
                axs[0].set_ylabel("Binary balanced accuracy")
                axs[-1].legend(frameon=False, fontsize=8, loc="lower right")
                fig.suptitle(f"Training seed {seed}"); fig.tight_layout(); pdf.savefig(fig)
                fig.savefig(root / f"relative_speed_{axis}_seed{seed}.png", dpi=140)
                plt.close(fig)
    # Class-specific recall for each specialist orientation.
    fig, axs = plt.subplots(3, 2, figsize=(10, 10), sharex=True, sharey=True)
    for r, version in enumerate(VERSIONS):
        for k, name in enumerate(CLASS_NAMES):
            ax = axs[r, k]
            for condition, color in zip(CONDITIONS, CONDITION_COLORS):
                g = summed[(summed.version == version) & (summed.condition == condition) & (summed.axis == "round")
                    & (summed.kind == "primary") & (summed.population == "all")].sort_values("cutoff")
                metric = f"recall_{k}"
                ax.plot(g.cutoff, g[f"{metric}_mean"], c=color, label=original.CONDITION_LABELS[condition])
                ax.fill_between(g.cutoff, g[f"{metric}_ci_low"], g[f"{metric}_ci_high"], color=color, alpha=.15)
            ax.axhline(1/2, c="gray", ls="--", lw=1); ax.set_ylim(0, 1)
            ax.set_title(f"{version}: {name}")
            ax.spines[["top", "right"]].set_visible(False)
            if k == 0: ax.set_ylabel("Class recall")
            if r == 2: ax.set_xlabel("Cumulative rounds")
    axs[-1, -1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(root / f"relative_speed_class_recall.{ext}", dpi=170)
    plt.close(fig)


def aggregate(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["root"])
    from run_relative_speed_representation import verify
    verify(manifest)
    tables, confusions, controls, embeddings = [], [], [], []
    for record in manifest["policies"]:
        out = Path(record["analysis_dir"])
        audit = json.loads((out / "completion.json").read_text())
        if audit["status"] != "completed" or audit["episodes"] != 920 or audit["class_counts"] != [460, 460]:
            raise ValueError(f"Incomplete binary policy analysis: {out}")
        if (audit["version"], audit["condition"], audit["seed"]) != (record["version"], record["condition"], record["seed"]):
            raise ValueError("Policy identity differs")
        table = pd.read_csv(out / "probe_per_seed.csv")
        if len(table) != 90:
            raise ValueError("Missing primary or shuffled binary probes")
        tables.append(table)
        confusions.append(pd.read_csv(out / "confusion_counts.csv"))
        controls.append(pd.read_csv(out / "random_baseline.csv"))
        embedding = pd.read_csv(out / "umap.csv")
        if len(embedding) != 920 or not np.isfinite(embedding[["umap_x", "umap_y"]]).all().all():
            raise ValueError("Invalid specialist-only UMAP")
        embedding["version"], embedding["condition"], embedding["training_seed"] = record["version"], record["condition"], record["seed"]
        embeddings.append(embedding)
    table = pd.concat(tables, ignore_index=True)
    combined_confusions = pd.concat(confusions, ignore_index=True)
    summed = summary(table)
    table.to_csv(root / "probe_per_seed.csv", index=False)
    summed.to_csv(root / "probe_summary.csv", index=False)
    combined_confusions.to_csv(root / "confusion_counts.csv", index=False)
    pd.concat(controls, ignore_index=True).to_csv(root / "random_baseline.csv", index=False)
    umap = pd.concat(embeddings, ignore_index=True)
    umap.to_csv(root / "umap_all_seeds.csv", index=False)
    endpoints = summed[(summed.kind == "primary") & (summed.population == "all") &
        (((summed.axis == "round") & (summed.cutoff == 20)) | ((summed.axis == "timestep") & (summed.cutoff == 400)))]
    endpoints.to_csv(root / "endpoints.csv", index=False)
    plots(table, summed, root)
    make_umap_book(umap, root)
    make_confusion_figure(combined_confusions, root)
    report = ["# Specialist-only relative-speed decoding", "",
        "v1_matched, v2, and v3; three RNN conditions; seeds 1–5: 45 fixed policies.", "",
        "Only the original 46 specialist profiles are used, with 20 episodes each: 920 episodes per policy. "
        "Red faster means d_R<d_B (class 0); blue faster means d_B<d_R (class 1). "
        "Equal-delay partners and all previous results containing them were deleted without archiving copies.", "",
        "Each profile contributes 16 fitting and four held-out episodes using the original PCG64 seed-0 repetition split. "
        "Class counts are 368/368 in fitting and 92/92 in testing (736/184 total). "
        "All 46 profiles occur in both probe partitions; this is held-out-episode decoding.", "",
        "Features are 128-D post-observation GRU states before the associated action: cumulative means through 20 round ends "
        "or timestep cutoffs 1,50,...,400. Terminal transitions are included and padding is excluded.", "",
        "Independent affine 128→2 softmax probes use full-batch Adam, lr=.01, 1000 updates, initialization seed 10000, "
        "no scaling/regularization/early stopping/test-based selection. Inverse-frequency weights are exactly 1 for primary probes "
        "because the two specialist classes are balanced. Primary metrics: binary accuracy, balanced accuracy, per-class recall, and confusion counts. Chance=1/2.", "",
        "All five policy seeds contribute to means and 95% percentile bootstrap intervals (10000 resamples, RNG 30000). "
        "Gaussian-feature controls retain five feature seeds; a shuffled-label control is fit at t=400 with RNG/init seed 20000.", "",
        "UMAP independently embeds each policy's 920 final-50-valid-state episode means from 128 dimensions to 2; "
        "Euclidean distance, n_neighbors=919, min_dist=1, spectral initialization, random_state=42. No labels are used during fitting.", "",
        "Reference plots: [prediction_fractions_state_comparison](prediction_fractions_state_comparison/README.md), "
        "pairing cumulative means and exact round-end states with two ground-truth panels and two prediction curves per panel. "
        "Each condition has comparison_all_versions.png and individual-seed PNGs.", "",
        "## Round-20 balanced accuracy", "", "| Experiment | Condition | Mean | 95% CI across seeds |", "|---|---|---:|---:|"]
    for row in endpoints[endpoints.axis == "round"].itertuples():
        report.append(f"| {row.version} | {original.CONDITION_LABELS[row.condition]} | {row.balanced_accuracy_mean:.3f} | [{row.balanced_accuracy_ci_low:.3f}, {row.balanced_accuracy_ci_high:.3f}] |")
    report.extend(["", "Per-policy models, predictions, episode manifests, controls, and UMAPs are under <version>/policies/<condition>_seed<N>/. "
                   "Aggregate tables/plots are in this root. Frozen sources and input hashes are in source/ and analysis_run.json. "
                   "No RL policy was retrained and no evaluation episodes were added.", ""])
    (root / "README.md").write_text("\n".join(report))
    write(root / "completion_audit.json", dict(status="completed", n_policies=45, n_episodes=41400,
        n_primary_probes=1305, n_umap_embeddings=45, episodes_per_policy=920,
        class_counts_per_policy=[460,460], original_rollouts_unchanged=True,
        completed_utc=datetime.now(timezone.utc).isoformat()))
    manifest["status"], manifest["completed_utc"] = "completed", datetime.now(timezone.utc).isoformat()
    write(manifest_path, manifest)
    print(f"[completed] {root}", flush=True)


def make_umap_book(table, root):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(root / "umap_relative_speed_all_seeds.pdf") as pdf:
        for version in VERSIONS:
            for seed in range(1, 6):
                fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
                for ax, condition in zip(axs, CONDITIONS):
                    draw_umap(ax, table[(table.version == version) & (table.training_seed == seed) & (table.condition == condition)])
                    ax.set_title(original.CONDITION_LABELS[condition])
                axs[-1].legend(frameon=False, fontsize=8)
                fig.suptitle(f"{version} · training seed {seed}"); fig.tight_layout(); pdf.savefig(fig)
                fig.savefig(root / f"umap_{version}_seed{seed}.png", dpi=140)
                plt.close(fig)


def make_confusion_figure(table, root):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 3, figsize=(12, 10))
    for r, version in enumerate(VERSIONS):
        for c, condition in enumerate(CONDITIONS):
            g = table[(table.version == version) & (table.condition == condition) & (table.population == "all")
                & (table.kind == "primary") & (table.axis == "round") & (table.cutoff == 20)]
            counts = g.groupby(["true_class", "predicted_class"])["count"].sum().unstack().to_numpy()
            normalized = counts / counts.sum(axis=1, keepdims=True)
            ax = axs[r, c]
            ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{normalized[i,j]:.2f}", ha="center", va="center",
                            color="white" if normalized[i,j] > .5 else "black")
            ax.set_xticks(range(2), CLASS_NAMES, rotation=20, fontsize=8)
            ax.set_yticks(range(2), CLASS_NAMES, fontsize=8)
            ax.set_title(f"{version}: {original.CONDITION_LABELS[condition]}", fontsize=10)
            ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    fig.suptitle("Round 20: row-normalized confusion, pooled across 5 seeds (equal test counts per seed)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(root / f"relative_speed_confusions.{ext}", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("analyze", "analyze-version", "aggregate"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("index", type=int, nargs="?")
    parser.add_argument("--version", choices=VERSIONS)
    args = parser.parse_args()
    if args.mode == "analyze":
        if args.index is None:
            parser.error("analyze requires a policy index")
        analyze(args.manifest, args.index)
    elif args.mode == "analyze-version":
        import time
        if args.version is None:
            parser.error("analyze-version requires --version")
        manifest = json.loads(args.manifest.read_text())
        for index, record in enumerate(manifest["policies"]):
            if record["version"] != args.version:
                continue
            analyze(args.manifest, index)
    else:
        aggregate(args.manifest)


if __name__ == "__main__":
    main()
