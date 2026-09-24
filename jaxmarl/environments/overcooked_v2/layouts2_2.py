import collections
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from itertools import product
import random
random.seed(42)

###############################################################################
# 1) Storing your skeleton layouts
###############################################################################
cramped_room = {
    "height" : 4,
    "width" : 5,
    "wall_idx" : jnp.array([0,1,2,3,4,
                            5,9,
                            10,14,
                            15,16,17,18,19]),
    "agent_idx" : jnp.array([6, 8]),
    "goal_idx" : jnp.array([18]),
    "plate_pile_idx" : jnp.array([16]),
    "onion_pile_idx" : jnp.array([5,9]),
    "tomato_pile_idx": jnp.array([]),
    "pot_idx" : jnp.array([2]),
    "recipe" : jnp.array([0]),
}

counter_circuit_v2 = """
WBPXW
WA  W
WWWWW
WA  W
WOTWW
"""


#############################################################
#############################################################
fivebyfive_1pots_2onion_1serving_1plates_open_1 = """
WWPWW
W   W
B   X
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_open_2 = """
WWPWW
W   W
X   B
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_open_3 = """
WWPWW
W   O
B   W
WA AO
WWXWW
"""

fivebyfive_1pots_2onion_1serving_1plates_open_4 = """
WWPWW
W   O
X   W
WA AO
WWBWW
"""

fivebyfive_1pots_2onion_1serving_1plates_open_5 = """
WWPWW
O   W
W   B
OA AW
WWXWW
"""

fivebyfive_1pots_2onion_1serving_1plates_open_6 = """
WWPWW
O   W
W   X
OA AW
WWBWW
"""

#############################################################
fivebyfive_2pots_2onion_1serving_1plates_open_1 = """
WWPPW
W   W
B   X
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_open_2 = """
WWPPW
W   W
X   B
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_open_3 = """
WWPPW
W   O
B   W
WA AO
WWXWW
"""

fivebyfive_2pots_2onion_1serving_1plates_open_4 = """
WWPPW
W   O
X   W
WA AO
WWBWW
"""

fivebyfive_2pots_2onion_1serving_1plates_open_5 = """
WWPPW
O   W
W   B
OA AW
WWXWW
"""

fivebyfive_2pots_2onion_1serving_1plates_open_6 = """
WWPPW
O   W
W   X
OA AW
WWBWW
"""


#############################################################
#############################################################
fivebyfive_1pots_2onion_1serving_1plates_closed_1 = """
WWPWW
WW WW
B   X
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_closed_2 = """
WWPWW
WW WW
X   B
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_closed_3 = """
WWPWW
WW OW
B   W
WA AO
WWXWW
"""

fivebyfive_1pots_2onion_1serving_1plates_closed_4 = """
WWPWW
WW OW
X   W
WA AO
WWBWW
"""

fivebyfive_1pots_2onion_1serving_1plates_closed_5 = """
WWPWW
WO WW
W   B
OA AW
WWXWW
"""

fivebyfive_1pots_2onion_1serving_1plates_closed_6 = """
WWPWW
WO WW
W   X
OA AW
WWBWW
"""

#############################################################
fivebyfive_2pots_2onion_1serving_1plates_closed_1 = """
WWPPW
WW WW
B   X
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_closed_2 = """
WWPPW
WW WW
X   B
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_closed_3 = """
WWPPW
WW OW
B   W
WA AO
WWXWW
"""

fivebyfive_2pots_2onion_1serving_1plates_closed_4 = """
WWPPW
WW OW
X   W
WA AO
WWBWW
"""

fivebyfive_2pots_2onion_1serving_1plates_closed_5 = """
WWPPW
WO WW
W   B
OA AW
WWXWW
"""

fivebyfive_2pots_2onion_1serving_1plates_closed_6 = """
WWPPW
WO WW
W   X
OA AW
WWBWW
"""

#############################################################
#############################################################
fivebyfive_1pots_2onion_1serving_1plates_divider_1 = """
WWWPW
W W W
B   X
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_divider_2 = """
WWWPW
W   W
X   B
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_divider_3 = """
WWWPW
W   O
B   W
WA AO
WWXWW
"""

fivebyfive_1pots_2onion_1serving_1plates_divider_4 = """
WWWPW
W   O
X   W
WA AO
WWBWW
"""

fivebyfive_1pots_2onion_1serving_1plates_divider_5 = """
WWWPW
O   W
W   B
OA AW
WWXWW
"""

fivebyfive_1pots_2onion_1serving_1plates_divider_6 = """
WWWPW
O   W
W   X
OA AW
WWBWW
"""

#############################################################
fivebyfive_2pots_2onion_1serving_1plates_divider_1 = """
WPWPW
W W W
B   X
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_divider_2 = """
WPWPW
W W W
X   B
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_divider_3 = """
WPWPW
W W O
B   W
WA AO
WWXWW
"""

fivebyfive_2pots_2onion_1serving_1plates_divider_4 = """
WPWPW
W W O
X   W
WA AO
WWBWW
"""

fivebyfive_2pots_2onion_1serving_1plates_divider_5 = """
WPWPW
O W W
W   B
OA AW
WWXWW
"""

fivebyfive_2pots_2onion_1serving_1plates_divider_6 = """
WPWPW
O W W
W   X
OA AW
WWBWW
"""

#############################################################
#############################################################

fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_1 = """
WWPWW
W   W
WB  X
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_2 = """
WWPWW
W   W
WX  B
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_3 = """
WWPWW
W   O
WB  W
WA AO
WWXWW
"""

fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_4 = """
WWPWW
W   O
WX  W
WA AO
WWBWW
"""

fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_5 = """
WWPWW
O   W
WW  B
OA AW
WWXWW
"""

fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_6 = """
WWPWW
O   W
WW  X
OA AW
WWBWW
"""

#############################################################
fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_1 = """
WWPPW
W   W
WB  X
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_2 = """
WWPPW
W   W
WX  B
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_3 = """
WWPPW
W   O
WB  W
WA AO
WWXWW
"""

fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_4 = """
WWPPW
W   O
WX  W
WA AO
WWBWW
"""

fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_5 = """
WWPPW
O   W
WW  B
OA AW
WWXWW
"""

fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_6 = """
WWPPW
O   W
WW  X
OA AW
WWBWW
"""


#############################################################
#############################################################

fivebyfive_1pots_2onion_1serving_1plates_island_1 = """
WWPWW
W   W
B W X
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_island_2 = """
WWPWW
W   W
X W B
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_island_3 = """
WWPWW
W   O
B W W
WA AO
WWXWW
"""

fivebyfive_1pots_2onion_1serving_1plates_island_4 = """
WWPWW
W   O
X W W
WA AO
WWBWW
"""

fivebyfive_1pots_2onion_1serving_1plates_island_5 = """
WWPWW
O   W
W W B
OA AW
WWXWW
"""

fivebyfive_1pots_2onion_1serving_1plates_island_6 = """
WWPWW
O   W
W W X
OA AW
WWBWW
"""

#############################################################
fivebyfive_2pots_2onion_1serving_1plates_island_1 = """
WWPPW
W   W
B W X
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_island_2 = """
WWPPW
W   W
X W B
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_island_3 = """
WWPPW
W   O
B W W
WA AO
WWXWW
"""

fivebyfive_2pots_2onion_1serving_1plates_island_4 = """
WWPPW
W   O
X W W
WA AO
WWBWW
"""

fivebyfive_2pots_2onion_1serving_1plates_island_5 = """
WWPPW
O   W
W W B
OA AW
WWXWW
"""

fivebyfive_2pots_2onion_1serving_1plates_island_6 = """
WWPPW
O   W
W W X
OA AW
WWBWW
"""

# layouts = {
#     "fivebyfive_1pots_2onion_1serving_1plates_open": """
# WWPWW
# W   W
# B   X
# WA AW
# WOWOW
# """,
#     "fivebyfive_2pots_2onion_1serving_1plates_open": """
# WWPPW
# W   W
# B   X
# WA AW
# WOWOW
# """,
#     "fivebyfive_1pots_2onion_1serving_1plates_closed": """
# WWPWW
# WW WW
# B   X
# WA AW
# WOWOW
# """,
#     "fivebyfive_2pots_2onion_1serving_1plates_closed": """
# WWPWW
# WW PW
# B   X
# WA AW
# WOWOW
# """,
#     "fivebyfive_2pots_2onion_1serving_1plates_divider": """
# WPWPW
# W W W
# B   X
# WA AW
# WOWOW
# """,
#     "fivebyfive_1pots_2onion_1serving_1plates_divider": """
# WWWPW
# W W W
# B   X
# WA AW
# WOWOW
# """,
#     "fivebyfive_1pots_2onion_1serving_1plates_island": """
# WWWPW
# B   W
# W W X
# WA AW
# WOWOW
# """,
#     "fivebyfive_2pots_2onion_1serving_1plates_island": """
# WWPPW
# B   W
# W W X
# WW AW
# WOWOW
# """
# }

###############################################################################
# Convert grid layout to dictionary
###############################################################################
def layout_grid_to_dict(grid):
    rows = grid.split('\n')
    if len(rows[0]) == 0:
        rows = rows[1:]
    if len(rows[-1]) == 0:
        rows = rows[:-1]
    keys = ["wall_idx", "agent_idx", "goal_idx", "plate_pile_idx", "onion_pile_idx", "tomato_pile_idx", "pot_idx"]
    symbol_to_key = {"W": "wall_idx",
                     "A": "agent_idx",
                     "X": "goal_idx",
                     "B": "plate_pile_idx",
                     "O": "onion_pile_idx",
                     "T": "tomato_pile_idx",
                     "P": "pot_idx"}
    layout_dict = {key: [] for key in keys}
    layout_dict["height"] = len(rows)
    layout_dict["width"] = len(rows[0])
    width = len(rows[0])
    for i, row in enumerate(rows):
        for j, obj in enumerate(row):
            idx = width * i + j
            if obj in symbol_to_key:
                layout_dict[symbol_to_key[obj]].append(idx)
            if obj in ["X", "B", "O", "T", "P"]:
                layout_dict["wall_idx"].append(idx)
            elif obj == " ":
                continue
    for key in symbol_to_key.values():
        layout_dict[key] = jnp.array(layout_dict[key])
    return FrozenDict(layout_dict)


swapped_results = {}
swapped_results['cramped_room'] = FrozenDict(cramped_room)
swapped_results['counter_circuit_v2'] = layout_grid_to_dict(counter_circuit_v2)
overcooked_v2_layouts = swapped_results


swapped_results.update({
    'fivebyfive_1pots_2onion_1serving_1plates_open_1': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_open_1),
    'fivebyfive_1pots_2onion_1serving_1plates_open_2': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_open_2),
    'fivebyfive_1pots_2onion_1serving_1plates_open_3': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_open_3),
    'fivebyfive_1pots_2onion_1serving_1plates_open_4': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_open_4),
    'fivebyfive_1pots_2onion_1serving_1plates_open_5': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_open_5),
    'fivebyfive_1pots_2onion_1serving_1plates_open_6': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_open_6),

    'fivebyfive_2pots_2onion_1serving_1plates_open_1': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_open_1),
    'fivebyfive_2pots_2onion_1serving_1plates_open_2': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_open_2),
    'fivebyfive_2pots_2onion_1serving_1plates_open_3': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_open_3),
    'fivebyfive_2pots_2onion_1serving_1plates_open_4': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_open_4),
    'fivebyfive_2pots_2onion_1serving_1plates_open_5': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_open_5),
    'fivebyfive_2pots_2onion_1serving_1plates_open_6': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_open_6),

    'fivebyfive_1pots_2onion_1serving_1plates_closed_1': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_closed_1),
    'fivebyfive_1pots_2onion_1serving_1plates_closed_2': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_closed_2),
    'fivebyfive_1pots_2onion_1serving_1plates_closed_3': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_closed_3),
    'fivebyfive_1pots_2onion_1serving_1plates_closed_4': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_closed_4),
    'fivebyfive_1pots_2onion_1serving_1plates_closed_5': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_closed_5),
    'fivebyfive_1pots_2onion_1serving_1plates_closed_6': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_closed_6),

    'fivebyfive_2pots_2onion_1serving_1plates_closed_1': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_closed_1),
    'fivebyfive_2pots_2onion_1serving_1plates_closed_2': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_closed_2),
    'fivebyfive_2pots_2onion_1serving_1plates_closed_3': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_closed_3),
    'fivebyfive_2pots_2onion_1serving_1plates_closed_4': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_closed_4),
    'fivebyfive_2pots_2onion_1serving_1plates_closed_5': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_closed_5),
    'fivebyfive_2pots_2onion_1serving_1plates_closed_6': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_closed_6),

    'fivebyfive_1pots_2onion_1serving_1plates_divider_1': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_divider_1),
    'fivebyfive_1pots_2onion_1serving_1plates_divider_2': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_divider_2),
    'fivebyfive_1pots_2onion_1serving_1plates_divider_3': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_divider_3),
    'fivebyfive_1pots_2onion_1serving_1plates_divider_4': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_divider_4),
    'fivebyfive_1pots_2onion_1serving_1plates_divider_5': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_divider_5),
    'fivebyfive_1pots_2onion_1serving_1plates_divider_6': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_divider_6),

    'fivebyfive_2pots_2onion_1serving_1plates_divider_1': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_divider_1),
    'fivebyfive_2pots_2onion_1serving_1plates_divider_2': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_divider_2),
    'fivebyfive_2pots_2onion_1serving_1plates_divider_3': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_divider_3),
    'fivebyfive_2pots_2onion_1serving_1plates_divider_4': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_divider_4),
    'fivebyfive_2pots_2onion_1serving_1plates_divider_5': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_divider_5),
    'fivebyfive_2pots_2onion_1serving_1plates_divider_6': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_divider_6),

    'fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_1': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_1),
    'fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_2': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_2),
    'fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_3': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_3),
    'fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_4': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_4),
    'fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_5': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_5),
    'fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_6': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_6),

    'fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_1': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_1),
    'fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_2': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_2),
    'fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_3': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_3),
    'fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_4': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_4),
    'fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_5': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_5),
    'fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_6': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_6),

    'fivebyfive_1pots_2onion_1serving_1plates_island_1': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_island_1),
    'fivebyfive_1pots_2onion_1serving_1plates_island_2': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_island_2),
    'fivebyfive_1pots_2onion_1serving_1plates_island_3': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_island_3),
    'fivebyfive_1pots_2onion_1serving_1plates_island_4': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_island_4),
    'fivebyfive_1pots_2onion_1serving_1plates_island_5': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_island_5),
    'fivebyfive_1pots_2onion_1serving_1plates_island_6': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_island_6),

    'fivebyfive_2pots_2onion_1serving_1plates_island_1': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_island_1),
    'fivebyfive_2pots_2onion_1serving_1plates_island_2': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_island_2),
    'fivebyfive_2pots_2onion_1serving_1plates_island_3': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_island_3),
    'fivebyfive_2pots_2onion_1serving_1plates_island_4': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_island_4),
    'fivebyfive_2pots_2onion_1serving_1plates_island_5': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_island_5),
    'fivebyfive_2pots_2onion_1serving_1plates_island_6': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_island_6),
})

overcooked_v2_layouts = swapped_results

# from process_layout_utils import convert_layout_str_to_grid
# for layout_name, layout_dict in overcooked_v2_layouts.items():
#     layout_grid = convert_layout_str_to_grid(str(layout_dict))
#     print(f"Layout: {layout_name}")    
#     for row in layout_grid:
#         print(row)
#     print()    


# def canonical_layout_repr(layout_dict):
#     rep = []
#     for key in sorted(layout_dict.keys()):
#         if key in ("height", "width"):
#             rep.append(f"{key}:{layout_dict[key]}")
#         else:
#             rep.append(f"{key}:{tuple(layout_dict[key].tolist())}")
#     return tuple(rep)

# unique_layouts = {}
# duplicates = []
# for layout_name, layout_dict in overcooked_v2_layouts.items():
#     rep = canonical_layout_repr(layout_dict)
#     if rep in unique_layouts:
#         duplicates.append((unique_layouts[rep], layout_name))
#     else:
#         unique_layouts[rep] = layout_name

# if duplicates:
#     print("Duplicates found:")
#     for dup in duplicates:
#         print("Duplicate layouts:", dup)
# else:
#     print("All layouts are unique!")
