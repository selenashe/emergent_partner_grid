# Pinned Overcooked performance reference

The upstream Apache license is preserved in `overcooked_LICENSE.txt`.

`overcooked_do_performance_plots.py.txt` preserves the Figure 2 analysis source
from the same upstream commit. Its SHA256 is
`435d953ac440d573f87f4291cb63d91fb00cdec55ab5cd54aa24a8ec17104903`.
It creates one reward row per episode, then calls `sns.pointplot` without
overriding its mean estimator, 95% confidence interval, or 1000 bootstrap draws.
There is no `units` argument or partner-profile stratification. The grid
adaptation is `eval/plot_overcooked_paper_match.py`; its outputs retain all
training candidates, selected seeds, episode returns, and source hashes.
