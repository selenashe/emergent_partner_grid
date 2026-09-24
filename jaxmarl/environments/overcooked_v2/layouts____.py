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


custom_layout = """
WWPWWPWW
WWA   WW
W    A W
WBXOOXBW
WWWWWWWW
"""


skeletons = {
    "counter_circuit": """
WWWWWWWW
W A    W
W WWWW W
W     AW
WWWWWWWW
""",
    "medium_room": """
WWWWWWWW
W A    W
W      W
W     AW
WWWWWWWW
""",
    "single_divider": """
WWWWWWWW
W A W  W
W   W  W
W     AW
WWWWWWWW
""",
    "double_divider": """
WWWWWWWW
W AWW  W
W  WW  W
W     AW
WWWWWWWW
"""
}

# Layout positions for pots, onions, servings, and plates
positions = {
    'P': [(0, 2), (0, 5)],  # Pot positions
    'O': [(4, 3), (4, 4)],  # Onion positions
    'B': [(4, 1), (4, 6)],  # Serving positions
    'X': [(4, 2), (4, 5)]   # Plate positions
}

layouts = {}
for skeleton_name, skeleton in skeletons.items():
    for perm in product(range(1, 3), repeat=4):  # Call product inside the loop
        grid = [list(row) for row in skeleton.strip().split('\n')]
        obj_counts = dict(zip(positions.keys(), perm))
        
        for obj, count in obj_counts.items():
            for i in range(count):
                x, y = positions[obj][i]
                grid[x][y] = obj
        
        layout_name = f"{skeleton_name}_{perm[0]}pots_{perm[1]}onion_{perm[2]}serving_{perm[3]}plates"
        layout_str = '\n'.join(''.join(row) for row in grid)
        layouts[layout_name] = layout_str


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
    "custom_layout": layout_grid_to_dict(custom_layout)
}

for name, layout_str in layouts.items():
    overcooked_v2_layouts[name] = layout_grid_to_dict(layout_str)