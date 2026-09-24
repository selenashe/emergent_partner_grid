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

##########################################################################
##########################################################################
counter_circuit_2pots_2onion_2serving_2plates = """
WXWWWWPW
X A    P
W WWWW W
O     AB
WOWWWWBW
"""

counter_circuit_1pots_1onion_1serving_1plates = """
WXWWWWWW
W A    P
W WWWW W
W     AB
WOWWWWWW
"""

counter_circuit_1pots_2onion_2serving_2plates = """
WXWWWWWW
X A    P
W WWWW W
O     AB
WOWWWWBW
"""

counter_circuit_2pots_1onion_2serving_2plates = """
WXWWWWPW
X A    P
W WWWW W
W     AB
WOWWWWBW
"""

counter_circuit_2pots_2onion_1serving_2plates = """
WXWWWWPW
W A    P
W WWWW W
O     AB
WOWWWWBW
"""

counter_circuit_2pots_2onion_2serving_1plates = """
WXWWWWPW
X A    P
W WWWW W
O     AB
WOWWWWWW
"""

counter_circuit_1pots_1onion_2serving_2plates = """
WXWWWWWW
X A    P
W WWWW W
W     AB
WOWWWWBW
"""

counter_circuit_1pots_2onion_1serving_2plates = """
WXWWWWWW
W A    P
W WWWW W
O     AB
WOWWWWBW
"""

counter_circuit_1pots_2onion_2serving_1plates = """
WXWWWWWW
X A    P
W WWWW W
O     AB
WOWWWWWW
"""

counter_circuit_2pots_1onion_1serving_2plates = """
WXWWWWPW
W A    P
W WWWW W
W     AB
WOWWWWBW
"""

counter_circuit_2pots_1onion_2serving_1plates = """
WXWWWWPW
X A    P
W WWWW W
W     AB
WOWWWWWW
"""

counter_circuit_2pots_2onion_1serving_1plates = """
WXWWWWPW
W A    P
W WWWW W
O     AB
WOWWWWWW
"""

counter_circuit_1pots_1onion_1serving_2plates = """
WXWWWWWW
W A    P
W WWWW W
W     AB
WOWWWWBW
"""

counter_circuit_1pots_1onion_2serving_1plates = """
WXWWWWWW
X A    P
W WWWW W
W     AB
WOWWWWWW
"""

counter_circuit_1pots_2onion_1serving_1plates = """
WXWWWWWW
W A    P
W WWWW W
O     AB
WOWWWWWW
"""

counter_circuit_2pots_1onion_1serving_1plates = """
WXWWWWPW
W A    P
W WWWW W
W     AB
WOWWWWWW
"""

##########################################################################
##########################################################################

medium_room_2pots_2onion_2serving_2plates = """
WXWWWWPW
X A    P
W      W
O     AB
WOWWWWBW
"""

medium_room_1pots_1onion_1serving_1plates = """
WXWWWWWW
W A    P
W      W
W     AB
WOWWWWWW
"""

medium_room_1pots_2onion_2serving_2plates = """
WXWWWWWW
X A    P
W      W
O     AB
WOWWWWBW
"""


medium_room_2pots_1onion_2serving_2plates = """
WXWWWWPW
X A    P
W      W
W     AB
WOWWWWBW
"""

medium_room_2pots_2onion_1serving_2plates = """
WXWWWWPW
W A    P
W      W
O     AB
WOWWWWBW
"""

medium_room_2pots_2onion_2serving_1plates = """
WXWWWWPW
X A    P
W      W
O     AB
WOWWWWWW
"""

medium_room_1pots_1onion_2serving_2plates = """
WXWWWWWW
X A    P
W      W
W     AB
WOWWWWBW
"""

medium_room_1pots_2onion_1serving_2plates = """
WXWWWWWW
W A    P
W      W
O     AB
WOWWWWBW
"""

medium_room_1pots_2onion_2serving_1plates = """
WXWWWWWW
X A    P
W      W
O     AB
WOWWWWWW
"""

medium_room_2pots_1onion_1serving_2plates = """
WXWWWWPW
W A    P
W      W
W     AB
WOWWWWBW
"""

medium_room_2pots_1onion_2serving_1plates = """
WXWWWWPW
X A    P
W      W
W     AB
WOWWWWWW
"""

medium_room_2pots_2onion_1serving_1plates = """
WXWWWWPW
W A    P
W      W
O     AB
WOWWWWWW
"""

medium_room_1pots_1onion_1serving_2plates = """
WXWWWWWW
W A    P
W      W
W     AB
WOWWWWBW
"""

medium_room_1pots_1onion_2serving_1plates = """
WXWWWWWW
X A    P
W      W
W     AB
WOWWWWWW
"""

medium_room_1pots_2onion_1serving_1plates = """
WXWWWWWW
W A    P
W      W
O     AB
WOWWWWWW
"""

medium_room_2pots_1onion_1serving_1plates = """
WXWWWWPW
W A    P
W      W
W     AB
WOWWWWWW
"""

##########################################################################
##########################################################################


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
    "counter_circuit_2pots_2onion_2serving_2plates" : layout_grid_to_dict(counter_circuit_2pots_2onion_2serving_2plates),
    "counter_circuit_1pots_1onion_1serving_1plates" : layout_grid_to_dict(counter_circuit_1pots_1onion_1serving_1plates),
    "counter_circuit_1pots_2onion_2serving_2plates" : layout_grid_to_dict(counter_circuit_1pots_2onion_2serving_2plates),
    "counter_circuit_2pots_1onion_2serving_2plates" : layout_grid_to_dict(counter_circuit_2pots_1onion_2serving_2plates),
    "counter_circuit_2pots_2onion_1serving_2plates" : layout_grid_to_dict(counter_circuit_2pots_2onion_1serving_2plates),
    "counter_circuit_2pots_2onion_2serving_1plates" : layout_grid_to_dict(counter_circuit_2pots_2onion_2serving_1plates),
    "counter_circuit_1pots_1onion_2serving_2plates" : layout_grid_to_dict(counter_circuit_1pots_1onion_2serving_2plates),
    "counter_circuit_1pots_2onion_1serving_2plates" : layout_grid_to_dict(counter_circuit_1pots_2onion_1serving_2plates),
    "counter_circuit_1pots_2onion_2serving_1plates" : layout_grid_to_dict(counter_circuit_1pots_2onion_2serving_1plates),
    "counter_circuit_2pots_1onion_1serving_2plates" : layout_grid_to_dict(counter_circuit_2pots_1onion_1serving_2plates),
    "counter_circuit_2pots_1onion_2serving_1plates" : layout_grid_to_dict(counter_circuit_2pots_1onion_2serving_1plates),
    "counter_circuit_2pots_2onion_1serving_1plates" : layout_grid_to_dict(counter_circuit_2pots_2onion_1serving_1plates),
    "counter_circuit_1pots_1onion_1serving_2plates" : layout_grid_to_dict(counter_circuit_1pots_1onion_1serving_2plates),
    "counter_circuit_1pots_1onion_2serving_1plates" : layout_grid_to_dict(counter_circuit_1pots_1onion_2serving_1plates),
    "counter_circuit_1pots_2onion_1serving_1plates" : layout_grid_to_dict(counter_circuit_1pots_2onion_1serving_1plates),
    "counter_circuit_2pots_1onion_1serving_1plates" : layout_grid_to_dict(counter_circuit_2pots_1onion_1serving_1plates),
    "medium_room_2pots_2onion_2serving_2plates" : layout_grid_to_dict(medium_room_2pots_2onion_2serving_2plates),
    "medium_room_1pots_1onion_1serving_1plates" : layout_grid_to_dict(medium_room_1pots_1onion_1serving_1plates),
    "medium_room_1pots_2onion_2serving_2plates" : layout_grid_to_dict(medium_room_1pots_2onion_2serving_2plates),
    "medium_room_2pots_1onion_2serving_2plates" : layout_grid_to_dict(medium_room_2pots_1onion_2serving_2plates),
    "medium_room_2pots_2onion_1serving_2plates" : layout_grid_to_dict(medium_room_2pots_2onion_1serving_2plates),
    "medium_room_2pots_2onion_2serving_1plates" : layout_grid_to_dict(medium_room_2pots_2onion_2serving_1plates),
    "medium_room_1pots_1onion_2serving_2plates" : layout_grid_to_dict(medium_room_1pots_1onion_2serving_2plates),
    "medium_room_1pots_2onion_1serving_2plates" : layout_grid_to_dict(medium_room_1pots_2onion_1serving_2plates),
    "medium_room_1pots_2onion_2serving_1plates" : layout_grid_to_dict(medium_room_1pots_2onion_2serving_1plates),
    "medium_room_2pots_1onion_1serving_2plates" : layout_grid_to_dict(medium_room_2pots_1onion_1serving_2plates),
    "medium_room_2pots_1onion_2serving_1plates" : layout_grid_to_dict(medium_room_2pots_1onion_2serving_1plates),
    "medium_room_2pots_2onion_1serving_1plates" : layout_grid_to_dict(medium_room_2pots_2onion_1serving_1plates),
    "medium_room_1pots_1onion_1serving_2plates" : layout_grid_to_dict(medium_room_1pots_1onion_1serving_2plates),
    "medium_room_1pots_1onion_2serving_1plates" : layout_grid_to_dict(medium_room_1pots_1onion_2serving_1plates),
    "medium_room_1pots_2onion_1serving_1plates" : layout_grid_to_dict(medium_room_1pots_2onion_1serving_1plates),
    "medium_room_2pots_1onion_1serving_1plates" : layout_grid_to_dict(medium_room_2pots_1onion_1serving_1plates)
}
