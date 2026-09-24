import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from itertools import product

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



fivebyfive_1pots_2onion_1serving_1plates_open = """
WWPWW
W   W
B   X
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_open = """
WWPPW
W   W
B   X
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_closed = """
WWPWW
WW WW
B   X
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_closed = """
WWPWW
WW PW
B   X
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_divider = """
WPWPW
W W W
B   X
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_divider = """
WWWPW
W W W
B   X
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_verticaldivider = """
WWWPW
B   W
WWW X
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_verticaldivider = """
WWPPW
B   W
WWW X
WA AW
WOWOW
"""


def layout_grid_to_dict(grid):
    """Assumes `grid` is string representation of the layout, with 1 line per row, and the following symbols:
    W: wall
    A: agent
    X: goal
    B: plate (bowl) pile
    O: onion pile
    T: tomato pile    
    P: pot location

    ' ' (space) : empty cell
    """

    rows = grid.split('\n')

    if len(rows[0]) == 0:
        rows = rows[1:]
    if len(rows[-1]) == 0:
        rows = rows[:-1]

    keys = ["wall_idx", "agent_idx", "goal_idx", "plate_pile_idx", "onion_pile_idx", "tomato_pile_idx", "pot_idx"]
    symbol_to_key = {"W" : "wall_idx",
                     "A" : "agent_idx",
                     "X" : "goal_idx",
                     "B" : "plate_pile_idx",
                     "O" : "onion_pile_idx",
                     "T" : "tomato_pile_idx",                     
                     "P" : "pot_idx"}

    layout_dict = {key : [] for key in keys}
    layout_dict["height"] = len(rows)
    layout_dict["width"] = len(rows[0])
    width = len(rows[0])

    for i, row in enumerate(rows):
        for j, obj in enumerate(row):
            idx = width * i + j
            if obj in symbol_to_key.keys():
                # Add object
                layout_dict[symbol_to_key[obj]].append(idx)
            if obj in ["X", "B", "O", "T", "P"]:
                # These objects are also walls technically
                layout_dict["wall_idx"].append(idx)
            elif obj == " ":
                # Empty cell
                continue

    for key in symbol_to_key.values():
        # Transform lists to arrays
        layout_dict[key] = jnp.array(layout_dict[key])

    return FrozenDict(layout_dict)

overcooked_v2_layouts = {
    "cramped_room": FrozenDict(cramped_room),
    "fivebyfive_1pots_2onion_1serving_1plates_open":  layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_open),
    "fivebyfive_2pots_2onion_1serving_1plates_open":  layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_open),
    "fivebyfive_1pots_2onion_1serving_1plates_closed":  layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_closed),
    "fivebyfive_2pots_2onion_1serving_1plates_closed":  layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_closed),
    "fivebyfive_2pots_2onion_1serving_1plates_divider":  layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_divider),
    "fivebyfive_1pots_2onion_1serving_1plates_divider":  layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_divider),
    "fivebyfive_1pots_2onion_1serving_1plates_verticaldivider":  layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_verticaldivider),
    "fivebyfive_2pots_2onion_1serving_1plates_verticaldivider":  layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_verticaldivider)
}

