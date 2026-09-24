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


five_0_A = """
WPWPW
O   O
W   W
BA AB
WXWXW
"""

five_1_A = """
WWWPWPW
W O   O
W     W
W BA AB
W  X XW
W     W
WWWWWWW
"""


five_2_A = """
WWWWPWPW
W  O   O
W      W
W  BA AB
W   X XW
W      W
W      W
WWWWWWWW
"""

five_3_A = """
WWWWWPWPW
W   O   O
W       W
W   BA AB
W    X XW
W       W
W       W
W       W
WWWWWWWWW
"""


five_4_A = """
WWWWWWPWPW
W    O   O
W        W
W    BA AB
W     X XW
W        W
W        W
W        W
W        W
WWWWWWWWWW
"""


five_5_A = """
WWWWWWWPWPW
W     O   O
W         W
W     BA AB
W      X XW
W         W
W         W
W         W
W         W
W         W
WWWWWWWWWWW
"""

five_6_A = """
WWWWWWWWPWPW
W      O   O
W          W
W      BA AB
W       X XW
W          W
W          W
W          W
W          W
W          W
W          W
WWWWWWWWWWWW
"""



five_0_B = """
WPWPW
O   O
W   W
BA AB
WXWXW
"""

five_1_B = """
WWWPWPW
W O   O
W W   W
W BA AB
W WXWXW
W     W
WWWWWWW
"""


five_2_B = """
WWWWPWPW
W  O   O
W  W   W
W  BA AB
W  WXWXW
W      W
W      W
WWWWWWWW
"""

five_3_B = """
WWWWWPWPW
W   O   O
W   W   W
W   BA AB
W   WXWXW
W       W
W       W
W       W
WWWWWWWWW
"""


five_4_B = """
WWWWWWPWPW
W    O   O
W    W   W
W    BA AB
W    WXWXW
W        W
W        W
W        W
W        W
WWWWWWWWWW
"""


five_5_B = """
WWWWWWWPWPW
W     O   O
W     W   W
W     BA AB
W     WXWXW
W         W
W         W
W         W
W         W
W         W
WWWWWWWWWWW
"""

five_6_B = """
WWWWWWWWPWPW
W      O   O
W      W   W
W      BA AB
W      WXWXW
W          W
W          W
W          W
W          W
W          W
W          W
WWWWWWWWWWWW
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
    "five_0_A": layout_grid_to_dict(five_0_A),
    "five_1_A": layout_grid_to_dict(five_1_A),
    "five_2_A": layout_grid_to_dict(five_2_A),
    "five_3_A": layout_grid_to_dict(five_3_A),
    "five_4_A": layout_grid_to_dict(five_4_A),
    "five_5_A": layout_grid_to_dict(five_5_A),
    "five_6_A": layout_grid_to_dict(five_6_A),
    "five_0_B": layout_grid_to_dict(five_0_B),
    "five_1_B": layout_grid_to_dict(five_1_B),
    "five_2_B": layout_grid_to_dict(five_2_B),
    "five_3_B": layout_grid_to_dict(five_3_B),
    "five_4_B": layout_grid_to_dict(five_4_B),
    "five_5_B": layout_grid_to_dict(five_5_B),
    "five_6_B": layout_grid_to_dict(five_6_B),
}
