from .coordination_grid import (
    CoordinationGrid,
    State,
    Actions,
    Allocations,
    Messages,           # backward-compat alias for Allocations
    encode_ego,
    decode_ego,
    N_MOVES,
    N_ALLOCATIONS,
    N_MESSAGES,         # backward-compat alias for N_ALLOCATIONS
    N_EGO_ACTIONS,
    GOAL_UNSET,
    GOAL_RED,
    GOAL_BLUE,
    LEGAL_ACTION_IDS_T0,
    LEGAL_ACTION_IDS_TGEQ1,
    ACTION_MASK_T0,
    ACTION_MASK_TGEQ1,
    COMM_ACTION_ONLY,
    COMM_CONDITIONS,
    bfs_distance_map,
)
from .capability_populations import (
    TRAIN_CAPABILITY_PAIRS,
    TEST_CAPABILITY_PAIRS,
    DEFAULT_SINGLE_PARTNER,
    FAST_DELAYS,
    SLOW_DELAYS,
)
