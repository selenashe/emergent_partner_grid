"""Locate descriptive return-divergence windows using existing training logs.

No training or checkpoint evaluation. Compare selected seeds within a condition;
for the two singleton v2 targets, use the median of the other nine same-condition
seeds. Persistence of overlapping rolling means is not a significance test.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnostics.select_debugging_seeds import CONDITIONS, HISTORY, LABELS, VERSIONS

SELECTION = ROOT / "train/seed_diagnostics/v1_matched_v2_v3/seed_selection"
DEFAULT_OUT = ROOT / "train/seed_diagnostics/v1_matched_v2_v3/return_divergence"
STEPS = 65536
COMPARISONS = (
    ("matched_rnn_1_vs_2", "v1_matched", "rnn_diverse_influence", 1, 2),
    ("matched_rnn_1_vs_4", "v1_matched", "rnn_diverse_influence", 1, 4),
    ("matched_rnn_4_vs_2", "v1_matched", "rnn_diverse_influence", 4, 2),
    ("matched_mlp_2_vs_8", "v1_matched", "mlp_diverse_influence", 2, 8),
    ("matched_mlp_10_vs_8", "v1_matched", "mlp_diverse_influence", 10, 8),
    ("matched_mlp_2_vs_10", "v1_matched", "mlp_diverse_influence", 2, 10),
    ("v3_rnn_10_vs_4", "v3", "rnn_diverse_influence", 10, 4),
    ("v3_single_6_vs_7", "v3", "rnn_single_influence", 6, 7),
    ("v2_rnn_10_vs_peers", "v2", "rnn_diverse_influence", 10, None),
    ("v2_single_8_vs_peers", "v2", "rnn_single_influence", 8, None),
)


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def key(row):
    return row["version"], row["condition"], int(row["seed"])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rolling(values, width, fraction=.75):
    valid = np.isfinite(values)
    count = np.convolve(valid.astype(int), np.ones(width), "valid")
    total = np.convolve(np.where(valid, values, 0), np.ones(width), "valid")
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count >= int(np.ceil(width*fraction)))
    return np.r_[np.full(width-1, np.nan), mean]


def other_median(values):
    matrix = np.array(values)
    usable = np.isfinite(matrix).sum(axis=0) >= 6
    result = np.full(matrix.shape[1], np.nan)
    result[usable] = np.nanmedian(matrix[:, usable], axis=0)
    return result


def segments(gap, threshold, persistence=8):
    result = []
    for sign in (1, -1):
        active = np.isfinite(gap) & (sign*gap >= threshold)
        boundaries = np.flatnonzero(np.diff(np.r_[False, active, False].astype(int)))
        for lo, hi in zip(boundaries[::2], boundaries[1::2]):
            if hi-lo >= persistence:
                result.append(dict(first_window_end_update=int(lo+1), last_window_end_update=int(hi),
                    confirmation_update=int(lo+persistence), direction="A higher" if sign == 1 else "B higher",
                    mean_signed_gap=float(gap[lo:hi].mean())))
    return sorted(result, key=lambda r: r["first_window_end_update"])


def plot(out, histories, target_keys):
    families = sorted({k[:2] for k in target_keys}, key=lambda p: (VERSIONS.index(p[0]), CONDITIONS.index(p[1])))
    colors = plt.get_cmap("tab10").colors
    x = np.arange(1, 916)*STEPS/1e6
    for limit, name in ((60, "return_curves_full"), (10, "return_curves_first_10M")):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
        for ax, (version, condition) in zip(axes.ravel(), families):
            ax.axvspan(0, 32*STEPS/1e6, color="#4477aa", alpha=.07)
            selected = sorted(k[2] for k in target_keys if k[:2] == (version, condition))
            for seed in range(1, 6):
                if seed not in selected:
                    ax.plot(x, rolling(histories[(version, condition, seed)], 8), color="#bbbbbb", alpha=.55, linewidth=.7)
            for seed in selected:
                ax.plot(x, rolling(histories[(version, condition, seed)], 8),
                    color=colors[seed-1], label=f"Seed {seed}", linewidth=1.8)
            if len(selected) == 1:
                ref = other_median([rolling(histories[(version, condition, seed)], 8) for seed in range(1, 6) if seed not in selected])
                ax.plot(x, ref, color="#222222", linestyle="--", linewidth=1.2, label="Other 9 median")
            ax.set_title(f"{version} · {LABELS[condition]}");ax.set_xlim(0, limit);ax.set_ylim(-17, 20)
            ax.grid(alpha=.15);ax.legend(fontsize=8, frameon=False, loc="lower right")
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes[1]:ax.set_xlabel("Training transitions (millions)")
        for ax in axes[:, 0]:ax.set_ylabel("Completed-episode return")
        fig.suptitle("Return histories of the 12 debugging targets", fontsize=16)
        fig.text(.5, .012, "8-update trailing means; missing episode returns excluded (at least 6/8 required). Gray: other seeds in the same condition.\n"
            "Shading: proposed initial dense capture through update 32. Curves are training returns, not held-out evaluations.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .06, 1, .95))
        fig.savefig(out/f"{name}.png", dpi=180);fig.savefig(out/f"{name}.pdf");plt.close(fig)


def plot_gap_onsets(out, summaries, curves):
    """Show the exact primary detector output separately for each comparison."""
    grouped = {r["comparison"]: [] for r in summaries}
    for r in curves:
        grouped[r["comparison"]].append(r)
    series = {name: np.array([[r["transitions"]/1e6, r["return_A"], r["return_B"]] for r in rows])
        for name, rows in grouped.items()}
    lasting = [r for r in summaries if r["terminal_gap_start"] is not None]
    colors = plt.get_cmap("tab10").colors
    markers = []
    for r in lasting:
        u = r["terminal_gap_start"]
        a = series[r["comparison"]]
        gap = a[:, 1]-a[:, 2]
        sign = 1 if r["terminal_gap_direction"] == "A higher" else -1
        assert np.all(np.isfinite(gap[u-1:])) and np.all(sign*gap[u-1:] >= 1.)
        assert u == 1 or not np.isfinite(gap[u-2]) or sign*gap[u-2] < 1.
        markers.append(dict(comparison=r["comparison"], version=r["version"], condition=r["condition"],
            seed_A=r["seed_A"], seed_B=r["seed_B"], onset_update=u, onset_transitions=u*STEPS,
            onset_million_transitions=u*STEPS/1e6, signed_gap_at_onset=float(gap[u-1]),
            confirmation_update=u+7, lasting_through_update=915, smoothing_updates=8, threshold=1.))
    write_csv(out/"lasting_gap_markers.csv", markers)

    def draw(ax, r, zoom):
        a = series[r["comparison"]]
        b = r["seed_B"]
        label_b = f"Seed {b}" if b != "median_other_9" else "Other 9 median"
        color_b = colors[int(b)-1] if b != "median_other_9" else "#202020"
        ax.plot(a[:, 0], a[:, 1], color=colors[r["seed_A"]-1], linewidth=2, label=f"Seed {r['seed_A']}")
        ax.plot(a[:, 0], a[:, 2], color=color_b, linewidth=2,
            linestyle=":" if b == "median_other_9" else "-", label=label_b)
        u = r["terminal_gap_start"]
        if u is not None:
            mark = u*STEPS/1e6
            ax.axvline(mark, color="#202020", linestyle="--", linewidth=1.5, zorder=4)
            ax.scatter([mark, mark], a[u-1, 1:3], color=[colors[r["seed_A"]-1], color_b], s=27, zorder=5)
            subtitle = f"Onset u{u} · {u*STEPS:,} transitions"
            if zoom:
                lo, hi = max(0, u-40), min(915, u+64)
                ax.set_xlim(lo*STEPS/1e6, hi*STEPS/1e6)
                local = a[max(0, lo-1):hi, 1:3]
                local = local[np.isfinite(local)]
                pad = max(.7, .12*np.ptp(local))
                ax.set_ylim(local.min()-pad, local.max()+pad)
            else:
                ax.set_xlim(0, 60);ax.set_ylim(-17, 20)
        else:
            subtitle = "No lasting ≥1-unit training-return gap"
            ax.set_xlim(0, 60);ax.set_ylim(-17, 20)
        ax.set_title(f"{r['version']} · {LABELS[r['condition']]} · seed {r['seed_A']} vs "
            f"{b if b != 'median_other_9' else 'peers'}\n{subtitle}", fontsize=11, pad=9)
        ax.set_xlabel("Training transitions (millions)", fontsize=10)
        ax.set_ylabel("Episode return", fontsize=10)
        ax.grid(alpha=.15);ax.legend(fontsize=9, frameon=False, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)

    for zoom, name in ((True, "lasting_gap_onsets_zoom"), (False, "lasting_gap_onsets_full")):
        fig, axes = plt.subplots(4, 2, figsize=(14, 15))
        assert len(lasting) == len(axes.ravel()) == 8
        for ax, r in zip(axes.ravel(), lasting):draw(ax, r, zoom)
        fig.suptitle("When the lasting return gap becomes visible", fontsize=18, y=.993)
        fig.text(.5, .968, "Each dashed line marks the exact primary-screen onset; the corresponding gap continues through update 915.",
            ha="center", fontsize=10)
        fig.text(.5, .012, "8-update trailing means · gap ≥1 return unit · ≥8 consecutive window endpoints · missing returns excluded\n"
            "" + ("Each panel zooms around its onset. " if zoom else "Full training horizon. ")
            + "These are descriptive detection times, not proven causal breakpoints.", ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .045, 1, .95), h_pad=2, w_pad=2)
        for ext in ("png", "pdf"):fig.savefig(out/f"{name}.{ext}", dpi=180)
        plt.close(fig)

    absent = [r for r in summaries if r["terminal_gap_start"] is None]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    assert len(absent) == 2
    for ax, r in zip(axes, absent):draw(ax, r, False)
    fig.suptitle("Two comparisons without a lasting training-return gap", fontsize=15)
    fig.text(.5, .015, "No onset line is assigned: early gaps close despite distinct final held-out behavior. Same primary rule as the marked panels.",
        ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .92))
    for ext in ("png", "pdf"):fig.savefig(out/f"no_lasting_training_return_gap.{ext}", dpi=180)
    plt.close(fig)
    return markers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args();out=args.out_dir.resolve();out.mkdir(parents=True, exist_ok=True)
    targets = {key(r) for r in read_csv(SELECTION/"representative_seeds.csv")}
    assert len(targets) == 12 and all(k[1] in CONDITIONS for k in targets)
    selection_audit = json.loads((SELECTION/"audit.json").read_text())
    source = HISTORY/"updates.csv"
    assert sha(source) == selection_audit["source_sha256"][str(source)]
    histories = {(v, c, s): np.full(915, np.nan) for v in VERSIONS for c in CONDITIONS for s in range(1, 6)}
    updates = {k: [] for k in histories}
    for r in read_csv(source):
        if key(r) not in histories:continue
        u=int(r["update"]);assert int(r["env_steps"]) == u*STEPS
        histories[key(r)][u-1]=float(r["episode_return"]);updates[key(r)].append(u)
    assert all(sorted(u) == list(range(1, 916)) for u in updates.values())
    summaries, events, curves = [], [], []
    for name, version, condition, a, b in COMPARISONS:
        raw_a = histories[(version, condition, a)]
        peers = [histories[(version, condition, seed)] for seed in range(1, 6) if seed != a]
        row = dict(comparison=name, version=version, condition=condition, seed_A=a,
            seed_B=str(b) if b is not None else "median_other_9", threshold=1., persistence=8,
            last32_A=float(np.nanmean(raw_a[-32:])),
            last32_B=float(np.nanmean(histories[(version, condition, b)][-32:])) if b is not None
                else float(np.median([np.nanmean(v[-32:]) for v in peers])))
        row["last32_signed_gap"] = row["last32_A"]-row["last32_B"]
        first_by_width, terminal_by_width = [], []
        for width in (4, 8, 16):
            sa=rolling(raw_a, width)
            sb=rolling(histories[(version, condition, b)], width) if b is not None else other_median([rolling(v, width) for v in peers])
            gap=sa-sb
            for threshold in (1., 2.):
                found=segments(gap, threshold)
                for event in found:
                    events.append(dict(comparison=name, width=width, threshold=threshold,
                        first_data_update=max(1, event["first_window_end_update"]-width+1), **event))
                if threshold == 1.:
                    if found:first_by_width.append(found[0]["first_window_end_update"])
                    terminal=next((e for e in found if e["last_window_end_update"] == 915), None)
                    if terminal:terminal_by_width.append(terminal["first_window_end_update"])
                    if width == 8:
                        first=found[0] if found else None
                        row.update(first_gap_update=first["first_window_end_update"] if first else None,
                            first_gap_direction=first["direction"] if first else "none",
                            first_gap_confirmation=first["confirmation_update"] if first else None,
                            terminal_gap_start=terminal["first_window_end_update"] if terminal else None,
                            terminal_gap_direction=terminal["direction"] if terminal else "none",
                            maximum_abs_gap_update=int(np.nanargmax(np.abs(gap))+1), maximum_abs_gap=float(np.nanmax(np.abs(gap))))
            if width == 8:
                for u in range(915):
                    curves.append(dict(comparison=name, update=u+1, transitions=(u+1)*STEPS,
                        return_A=sa[u], return_B=sb[u], signed_gap=gap[u]))
        row.update(first_onset_min=min(first_by_width) if first_by_width else None,
            first_onset_max=max(first_by_width) if first_by_width else None,
            terminal_onset_min=min(terminal_by_width) if terminal_by_width else None,
            terminal_onset_max=max(terminal_by_width) if terminal_by_width else None)
        summaries.append(row)
    existing = {key(r): r for r in read_csv(ROOT/"train/seed_diagnostics/v1_matched_v2_v3/per_seed.csv")}
    target_stats=[]
    for identity in sorted(targets):
        values=histories[identity];mean=rolling(values, 46, fraction=1.);valid=np.flatnonzero(np.isfinite(mean))
        drawdown=np.maximum.accumulate(mean[valid])-mean[valid];i=int(np.argmax(drawdown))
        maximum=float(drawdown[i])
        assert np.isclose(maximum, float(existing[identity]["maximum46_drawdown"]), atol=1e-10)
        target_stats.append(dict(version=identity[0], condition=identity[1], seed=identity[2],
            finite_return_updates=int(np.isfinite(values).sum()), first_finite_update=int(np.flatnonzero(np.isfinite(values))[0]+1),
            maximum46_drawdown=maximum, drawdown_end_update=int(valid[i]+1),
            final46_return=float(mean[-1]), best46_return=float(np.nanmax(mean))))
    write_csv(out/"comparisons.csv", summaries);write_csv(out/"gap_intervals.csv", events)
    write_csv(out/"comparison_curves.csv", curves);write_csv(out/"target_return_audit.csv", target_stats)
    plot(out, histories, targets)
    markers=plot_gap_onsets(out, summaries, curves)
    audit=dict(selected_policies=12, available_reference_policies=45, updates_per_policy=915,
        full_training_history_scanned=True, new_training_launched=False,
        primary_rule="Absolute 8-update trailing-mean gap >=1 return unit for >=8 consecutive window endpoints; at least 6/8 finite returns per seed/window.",
        sensitivity="Also 4- and 16-update windows (>=75% finite) and threshold 2; these are descriptive screens, not confidence intervals or hypothesis tests.",
        singleton_reference="Median of other nine same-version, same-condition smoothed histories; require at least six finite peer histories.",
        terminal_rule="The threshold-qualified interval reaches update 915; distinct from an early temporary gap.",
        onset_plot_markers=markers,
        caveats=["Completed-episode returns span multiple policy updates and can lag the behavior that changed.",
            "Early missing returns mean no completed episode was logged; they are not zero rewards.",
            "Overlapping rolling windows are correlated; persistence does not establish statistical significance.",
            "Different color preferences or held-out failures may have similar training returns.",
            "Onsets describe logged return separation, not exact causal optimizer or transition timestamps."],
        source_sha256={str(p):sha(p) for p in (Path(__file__), source, SELECTION/"representative_seeds.csv", SELECTION/"audit.json")})
    (out/"audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(dict(comparisons=summaries, maximum46_drawdown_among_targets=max(r["maximum46_drawdown"] for r in target_stats), output=str(out)), indent=2))


if __name__ == "__main__":
    main()
