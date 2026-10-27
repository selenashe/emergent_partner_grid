"""Centralized capability populations for the CoordinationGrid experiment.

Follows the Overcooked reference (Mon-Williams et al) convention:
partner capability is a pair ``(d_R, d_B)`` of per-goal *delays* where
``d_X`` is the number of wait steps between successive partner moves
while pursuing goal X. So

    d = 0  -> partner moves every step   (fastest)
    d = 1  -> partner moves every 2 steps
    d = k  -> partner moves every (k + 1) steps

Training partners are "specialists": low delay in one task, high delay
in the other. This is the pressure the ego needs to discover.

Test partners are held-out capability profiles built from scalar delay
values NEVER seen during training (0, 5, 6 — the reference's "novel"
values), so the split tests generalization to genuinely new capability
values, not merely unseen combinations of familiar values.

Only ``TRAIN_CAPABILITY_PAIRS`` and ``TEST_CAPABILITY_PAIRS`` are
authoritative here. Anywhere else in the codebase these constants
should be imported from this module.
"""

from __future__ import annotations

from itertools import product
from typing import Tuple

# ---- Scalar delay pools ----------------------------------------------------
# "Fast" / low-delay partners on a task; "slow" / high-delay partners.
FAST_DELAYS: Tuple[int, ...] = (1, 2, 3)
SLOW_DELAYS: Tuple[int, ...] = (4, 7, 8, 9)

# Training specialist pairs: each partner is relatively fast at ONE task
# and relatively slow at the OTHER. Balanced across orientation:
#   (fast RED, slow BLUE) union (slow RED, fast BLUE)
TRAIN_CAPABILITY_PAIRS: Tuple[Tuple[int, int], ...] = tuple(
    [(dr, db) for dr in FAST_DELAYS for db in SLOW_DELAYS]
    + [(dr, db) for dr in SLOW_DELAYS for db in FAST_DELAYS]
)  # |FAST| * |SLOW| * 2 = 3 * 4 * 2 = 24 pairs

# Novel test partners include scalar values NEVER seen during training.
# Structure mirrors the reference `speed_pairs_test`.
TEST_NOVEL_LOW: Tuple[int, ...] = (5, 6)   # novel "mid" delays
TEST_NOVEL_ZERO: Tuple[int, ...] = (0,)    # novel "instant" delay
TEST_CAPABILITY_PAIRS: Tuple[Tuple[int, int], ...] = tuple(
    list(product((0, 1, 2, 3), TEST_NOVEL_LOW))       # fast/novel-low RED x novel-mid BLUE
    + list(product((7, 8, 9),  TEST_NOVEL_ZERO))      # slow RED x novel-zero BLUE
    + list(product(TEST_NOVEL_LOW,   (0, 1, 2, 3)))   # novel-mid RED x fast/novel-low BLUE
    + list(product(TEST_NOVEL_ZERO,  (7, 8, 9)))      # novel-zero RED x slow BLUE
)  # 8 + 3 + 8 + 3 = 22 pairs

# Default single-partner profile for the single-partner-RNN control.
# Chosen deterministically as a mid-range specialist so both tasks have
# meaningfully different completion times. Not tuned against eval.
DEFAULT_SINGLE_PARTNER: Tuple[int, int] = (1, 4)


def _validate() -> None:
    train_set = set(TRAIN_CAPABILITY_PAIRS)
    test_set = set(TEST_CAPABILITY_PAIRS)
    assert train_set.isdisjoint(test_set), (
        "TRAIN_CAPABILITY_PAIRS and TEST_CAPABILITY_PAIRS must be disjoint"
    )
    train_scalars = set(v for pair in TRAIN_CAPABILITY_PAIRS for v in pair)
    test_scalars = set(v for pair in TEST_CAPABILITY_PAIRS for v in pair)
    truly_novel = test_scalars - train_scalars
    assert truly_novel >= {0, 5, 6}, (
        f"test scalars must include the novel values {{0,5,6}}; missing "
        f"{{0,5,6}} - {truly_novel}"
    )
    for d in list(FAST_DELAYS) + list(SLOW_DELAYS):
        assert d >= 0, f"delay must be >= 0; got {d}"


_validate()
