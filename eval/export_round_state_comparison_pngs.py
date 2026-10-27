"""Export one all-version comparison PNG per RNN condition from saved scores."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def export(root):
    root = Path(root).resolve()
    source = root / "source/round_state_prediction_comparison.py"
    spec = importlib.util.spec_from_file_location("saved_round_state_plotting", source)
    plotting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotting)
    table_path = root / "prediction_fractions_summary.csv"
    table = pd.read_csv(table_path)
    outputs = []
    for condition, name in plotting.CONDITIONS.items():
        fig, axes = plt.subplots(6, 2, figsize=(10, 22), sharey=True, sharex=True)
        fig.suptitle(f"{name}\nAll 5 training seeds · shading: 95% bootstrap CI across seeds", fontsize=15, y=.99)
        for version_index, version in enumerate(plotting.VERSIONS):
            for feature_index, feature in enumerate(plotting.FEATURE_NAMES):
                row = version_index * 2 + feature_index
                selected = table[(table.version == version) & (table.condition == condition)
                                 & (table.feature_type == feature)]
                for col, actual in enumerate(plotting.PANEL_ORDER):
                    panel = selected[selected.ground_truth_class == actual]
                    if len(panel) != 40:
                        raise ValueError("Expected two prediction curves at all twenty rounds")
                    plotting.panel(axes[row, col], selected, actual, aggregated=True)
                    axes[row, col].set_title(
                        f"{version} · Ground truth: {plotting.CLASS_NAMES[actual]}\n"
                        f"({plotting.CLASS_DELAYS[actual]})", fontsize=11)
                    axes[row, col].tick_params(labelbottom=feature_index == 1)
                    if feature_index == 0:
                        axes[row, col].set_xlabel("")
                axes[row, 0].set_ylabel(f"{plotting.FEATURE_NAMES[feature]}\nFraction predicted as each class", fontsize=10)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .02), ncol=2, frameon=False, fontsize=12)
        fig.text(.5, .009, "Each version: cumulative mean above, single round-end state below. "
                 "Two solid lines sum to 1 within each panel; dashed guide = 1/2.", ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .045, 1, .965), h_pad=2.1, w_pad=1.6)
        destination = root / condition / "comparison_all_versions.png"
        fig.savefig(destination, dpi=180)
        plt.close(fig)
        outputs.append(dict(condition=condition, path=str(destination), width=1800, height=3960,
                            sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))
        print(destination, flush=True)
    script = Path(__file__).resolve()
    frozen = root / "source" / script.name
    if script != frozen:
        shutil.copyfile(script, frozen)
    metadata = dict(status="completed", exported_utc=datetime.now(timezone.utc).isoformat(),
                    input_summary=str(table_path), input_summary_sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(),
                    plotting_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    export_source_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                    layout="six rows by two ground-truth columns; each version occupies two rows (cumulative mean, discrete state)",
                    outputs=outputs)
    (root / "all_versions_png_export.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    export(parser.parse_args().root)
