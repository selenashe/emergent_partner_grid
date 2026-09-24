import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
import random


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


########################################
fivebyfive_1pots_2onion_1serving_1plates_open_a = """
WWPWW
W   W
B   X
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_open_b = """
WWWPW
W   W
B   X
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_open_c = """
WWPWW
W   W
X   B
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_open_d = """
WWWPW
W   W
X   B
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_open_a = """
WWPPW
W   W
B   X
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_open_b = """
WPPWW
W   W
B   X
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_open_c = """
WWPPW
W   W
X   B
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_open_d = """
WPPWW
W   W
X   B
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_closed_a = """
WWPWW
WW WW
B   X
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_closed_b = """
WWPWW
WW WW
X   B
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_closed_c = """
WWPWW
WW WW
WB  X
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_closed_d = """
WWPWW
WW WW
WX  B
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_closed_a = """
WWPWW
WW PW
B   X
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_closed_b = """
WWPWW
WP WW
B   X
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_closed_c = """
WWPWW
WW PW
X   B
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_closed_d = """
WWPWW
WP WW
X   B
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_divider_a = """
WWWPW
W W W
B   X
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_divider_b = """
WPWWW
W W W
B   X
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_divider_c = """
WWWPW
W W W
X   B
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_divider_d = """
WPWWW
W W W
X   B
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_divider_a = """
WPWPW
W W W
B   X
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_divider_b = """
WWWWW
P W P
B   X
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_divider_c = """
WPWPW
W W W
X   B
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_divider_d = """
WWWWW
P W P
X   B
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_a = """
WWWPW
B   W
WWW X
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_b = """
WWPWW
B   W
WWW X
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_c = """
WWWPW
X   W
WWW B
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_d = """
WWPWW
X   W
WWW B
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_a = """
WWPPW
B   W
WWW X
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_b = """
WPPWW
B   W
WWW X
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_c = """
WWPPW
X   W
WWW B
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_d = """
WPPWW
X   W
WWW B
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_longdivider_a = """
WPWWW
W W W
B W X
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_longdivider_b = """
WWWPW
W W W
B W X
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_longdivider_c = """
WPWWW
W W W
X W B
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_longdivider_d = """
WWWPW
W W W
X W B
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_longdivider_a = """
WPWPW
W W W
B W X
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_longdivider_b = """
WWWWW
P W P
B W X
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_longdivider_c = """
WPWPW
W W W
X W B
WA AW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_longdivider_d = """
WWWWW
P W P
X W B
WA AW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_twodividers_a = """
WWWPW
W W W
X   B
WAWAW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_twodividers_b = """
WPWWW
W W W
X   B
WAWAW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_twodividers_c = """
WWWPW
W W W
B   X
WAWAW
WOWOW
"""

fivebyfive_1pots_2onion_1serving_1plates_twodividers_d = """
WPWWW
W W W
B   X
WAWAW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_twodividers_a = """
WPWPW
W W W
X   B
WAWAW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_twodividers_b = """
WPWPW
W W W
B   X
WAWAW
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_twodividers_c = """
WPWPW
W W W
W   W
XAWAB
WOWOW
"""

fivebyfive_2pots_2onion_1serving_1plates_twodividers_d = """
WPWPW
W W W
W   W
BAWAX
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

swapped_results = {}
swapped_results['cramped_room'] = FrozenDict(cramped_room)

swapped_results.update({
'fivebyfive_1pots_2onion_1serving_1plates_open_a': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_open_a),
'fivebyfive_1pots_2onion_1serving_1plates_open_b': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_open_b),
'fivebyfive_1pots_2onion_1serving_1plates_open_c': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_open_c),
'fivebyfive_1pots_2onion_1serving_1plates_open_d': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_open_d),
'fivebyfive_2pots_2onion_1serving_1plates_open_a': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_open_a),
'fivebyfive_2pots_2onion_1serving_1plates_open_b': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_open_b),
'fivebyfive_2pots_2onion_1serving_1plates_open_c': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_open_c),
'fivebyfive_2pots_2onion_1serving_1plates_open_d': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_open_d),
'fivebyfive_1pots_2onion_1serving_1plates_closed_a': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_closed_a),
'fivebyfive_1pots_2onion_1serving_1plates_closed_b': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_closed_b),
'fivebyfive_1pots_2onion_1serving_1plates_closed_c': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_closed_c),
'fivebyfive_1pots_2onion_1serving_1plates_closed_d': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_closed_d),
'fivebyfive_2pots_2onion_1serving_1plates_closed_a': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_closed_a),
'fivebyfive_2pots_2onion_1serving_1plates_closed_b': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_closed_b),
'fivebyfive_2pots_2onion_1serving_1plates_closed_c': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_closed_c),
'fivebyfive_2pots_2onion_1serving_1plates_closed_d': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_closed_d),
'fivebyfive_1pots_2onion_1serving_1plates_divider_a': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_divider_a),
'fivebyfive_1pots_2onion_1serving_1plates_divider_b': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_divider_b),
'fivebyfive_1pots_2onion_1serving_1plates_divider_c': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_divider_c),
'fivebyfive_1pots_2onion_1serving_1plates_divider_d': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_divider_d),
'fivebyfive_2pots_2onion_1serving_1plates_divider_a': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_divider_a),
'fivebyfive_2pots_2onion_1serving_1plates_divider_b': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_divider_b),
'fivebyfive_2pots_2onion_1serving_1plates_divider_c': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_divider_c),
'fivebyfive_2pots_2onion_1serving_1plates_divider_d': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_divider_d),
'fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_a': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_a),
'fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_b': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_b),
'fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_c': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_c),
'fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_d': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_verticaldivider_d),
'fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_a': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_a),
'fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_b': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_b),
'fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_c': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_c),
'fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_d': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_verticaldivider_d),
'fivebyfive_1pots_2onion_1serving_1plates_longdivider_a': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_longdivider_a),
'fivebyfive_1pots_2onion_1serving_1plates_longdivider_b': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_longdivider_b),
'fivebyfive_1pots_2onion_1serving_1plates_longdivider_c': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_longdivider_c),
'fivebyfive_1pots_2onion_1serving_1plates_longdivider_d': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_longdivider_d),
'fivebyfive_2pots_2onion_1serving_1plates_longdivider_a': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_longdivider_a),
'fivebyfive_2pots_2onion_1serving_1plates_longdivider_b': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_longdivider_b),
'fivebyfive_2pots_2onion_1serving_1plates_longdivider_c': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_longdivider_c),
'fivebyfive_2pots_2onion_1serving_1plates_longdivider_d': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_longdivider_d),
'fivebyfive_1pots_2onion_1serving_1plates_twodividers_a': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_twodividers_a),
'fivebyfive_1pots_2onion_1serving_1plates_twodividers_b': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_twodividers_b),
'fivebyfive_1pots_2onion_1serving_1plates_twodividers_c': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_twodividers_c),
'fivebyfive_1pots_2onion_1serving_1plates_twodividers_d': layout_grid_to_dict(fivebyfive_1pots_2onion_1serving_1plates_twodividers_d),
'fivebyfive_2pots_2onion_1serving_1plates_twodividers_a': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_twodividers_a),
'fivebyfive_2pots_2onion_1serving_1plates_twodividers_b': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_twodividers_b),
'fivebyfive_2pots_2onion_1serving_1plates_twodividers_c': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_twodividers_c),
'fivebyfive_2pots_2onion_1serving_1plates_twodividers_d': layout_grid_to_dict(fivebyfive_2pots_2onion_1serving_1plates_twodividers_d),
})


overcooked_v2_layouts = swapped_results