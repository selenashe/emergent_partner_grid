"""Check category weighting and uncertainty over independent policy seeds."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import plot_capability_category_decoding as category


def test_profile_averages_are_computed_within_seed_before_summarizing():
    root = Path(__file__).resolve().parents[2] / "eval/evaluation_plans/t0fix_direct_input_capability_review_20261022"
    profiles = category.read(root / "capability_review.json")["proposed_profiles"]
    rows = []
    for version in category.VERSIONS:
        for condition in category.CONDITIONS:
            for seed in range(1, 6):
                for profile in profiles:
                    red, blue = profile["red_delay"], profile["blue_delay"]
                    correct = (seed + red + blue) % 5
                    for cutoff in category.TIMES:
                        rows.append(dict(version=version, condition=condition, training_seed=seed,
                            d_R=red, d_B=blue, cutoff=cutoff, true_class=int(blue < red),
                            n_episodes=4, n_correct=correct, accuracy=correct / 4))
    table = pd.DataFrame(rows)
    per_seed, summary = category.category_averages(table, profiles)
    assert len(per_seed) == 1620 and len(summary) == 324
    selected = per_seed[(per_seed.version == "v2") & (per_seed.condition == category.CONDITIONS[0])
                        & (per_seed.category == "both_delays_novel") & (per_seed.cutoff == 50)].sort_values("training_seed")
    expected = np.array([np.mean([(seed + p["red_delay"] + p["blue_delay"]) % 5 / 4
                                 for p in profiles if p["category"] == "both_delays_novel"])
                         for seed in range(1, 6)])
    np.testing.assert_allclose(selected.accuracy, expected)
    assert set(selected.n_profiles) == {6} and set(selected.n_episodes) == {24}
    row = summary[(summary.version == "v2") & (summary.condition == category.CONDITIONS[0])
                  & (summary.category == "both_delays_novel") & (summary.cutoff == 50)].iloc[0]
    bootstrap = expected[np.random.default_rng(30000).integers(0, 5, (10000, 5))].mean(axis=1)
    assert row.accuracy_mean == pytest.approx(expected.mean())
    assert row.accuracy_ci_low == pytest.approx(np.quantile(bootstrap, .025))
    assert row.accuracy_ci_high == pytest.approx(np.quantile(bootstrap, .975))
    with pytest.raises(ValueError, match="Incomplete decoding table"):
        category.category_averages(table.iloc[:-1], profiles)


def test_bootstrap_uses_five_seed_means_independently_of_profile_count():
    base = dict(version="v3", condition=category.CONDITIONS[0], cutoff=400)
    frames = []
    for name, count in [("one_delay_novel", 42), ("both_delays_novel", 6)]:
        frames.extend(dict(**base, category=name, training_seed=seed, accuracy=float(seed == 5),
                           n_profiles=count, n_episodes=4 * count) for seed in range(1, 6))
    table = pd.DataFrame(frames)
    summary = category.bootstrap_seeds(table)
    assert summary.accuracy_mean.nunique() == 1
    assert summary.accuracy_ci_low.nunique() == summary.accuracy_ci_high.nunique() == 1
    with pytest.raises(ValueError, match="five policy seeds"):
        category.bootstrap_seeds(table.iloc[:-1])
