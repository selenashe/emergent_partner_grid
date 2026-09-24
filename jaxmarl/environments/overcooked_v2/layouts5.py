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
asymm_advantages = {
    "height" : 5,
    "width" : 9,
    "wall_idx" : jnp.array([0,1,2,3,4,5,6,7,8,
                            9,11,12,13,14,15,17,
                            18,22,26,
                            27,31,35,
                            36,37,38,39,40,41,42,43,44]),
    "agent_idx" : jnp.array([29, 32]),
    "goal_idx" : jnp.array([12,17]),
    "plate_pile_idx" : jnp.array([39,41]),
    "onion_pile_idx" : jnp.array([9,14]),
    "tomato_pile_idx": jnp.array([]),    
    "pot_idx" : jnp.array([22,31]),
}
coord_ring = {
    "height" : 5,
    "width" : 5,
    "wall_idx" : jnp.array([0,1,2,3,4,
                            5,9,
                            10,12,14,
                            15,19,
                            20,21,22,23,24]),
    "agent_idx" : jnp.array([7, 11]),
    "goal_idx" : jnp.array([22]),
    "plate_pile_idx" : jnp.array([10]),
    "onion_pile_idx" : jnp.array([15,21]),
    "tomato_pile_idx": jnp.array([]),
    "pot_idx" : jnp.array([3,9]),
}
forced_coord = {
    "height" : 5,
    "width" : 5,
    "wall_idx" : jnp.array([0,1,2,3,4,
                            5,7,9,
                            10,12,14,
                            15,17,19,
                            20,21,22,23,24]),
    "agent_idx" : jnp.array([11,8]),
    "goal_idx" : jnp.array([23]),
    "onion_pile_idx" : jnp.array([5,10]),
    "plate_pile_idx" : jnp.array([15]),
    "tomato_pile_idx": jnp.array([]),    
    "pot_idx" : jnp.array([3,9]),
}
cramped_room_tomatoes = {
    "height" : 4,
    "width" : 5,
    "wall_idx" : jnp.array([0,1,2,3,4,
                            5,9,
                            10,14,
                            15,16,17,18,19]),
    "agent_idx" : jnp.array([6, 8]),
    "goal_idx" : jnp.array([18]),
    "onion_pile_idx" : jnp.array([5]),
    "plate_pile_idx" : jnp.array([16]),
    "tomato_pile_idx": jnp.array([9]),
    "pot_idx" : jnp.array([2]),
}
asymm_advantages_tomatoes = {
    "height" : 5,
    "width" : 9,
    "wall_idx" : jnp.array([0,1,2,3,4,5,6,7,8,
                            9,11,12,13,14,15,17,
                            18,22,26,
                            27,31,35,
                            36,37,38,39,40,41,42,43,44]),
    "agent_idx" : jnp.array([29, 32]),
    "goal_idx" : jnp.array([12,17]),
    "plate_pile_idx" : jnp.array([39,41]),
    "onion_pile_idx" : jnp.array([9]),
    "tomato_pile_idx": jnp.array([14]),    
    "pot_idx" : jnp.array([22,31]),
}
# Example of layout provided as a grid
counter_circuit_grid = """
WWWPPWWW
W A    W
B WWWW X
W     AW
WWWOOWWW
"""

bottleneck_room_grid = """
WBWWWTW
WA W  W
P   A W
X W  OW
WPWWWWW
"""

big_room = """
WWWPWWW
OA   AO
W     W
W     W
W     W
WBWWWXW
"""

L_shaped = """
WXPBWT
TA
WA
O
"""

U_shaped = """
WWXBW
TA  W
OA  P
W   W
"""

gallery = """
WXPBWO
    A 
  A   
WWWWWT
"""

single_wall = """
POWBXTW
A      
 A     
"""

too_many_cooks = """
WWWW
XAWW
P WW
B  W
OAWW
WWWW
"""

signaling_room = """
WWWWWWW
W     W
W W W W
O W  AW
X W   P
T W  AB
W W W W
W     W
WWWWWWW
"""

semi_forced_coord = """
WWWWW
WA  W
W   W
WAW W
O W P
B W W
WWWXW
"""

two_islands = """
WWXPWWW
WA    W
O  W  T
W     W
W  W  W
WA    W
WWBPXWW
"""

H_shape = """
WWWWWWW
WA W  T
O     X
WA W  B
WWWWPWW
"""


###################################################


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

def generate_variants(base_layout, max_variants=100):
    """Generates layout variants by choosing unique positions for items from the original wall_idx."""
    variants = []

    wall_positions = base_layout["wall_idx"].tolist() 
    agent_positions = set(base_layout["agent_idx"].tolist())  
    available_positions = list(set(wall_positions) - agent_positions) # Possible positions for objects

    # Types and numbers of objects to place in layout
    items = {
        "goal_idx": base_layout["goal_idx"].tolist(),
        "plate_pile_idx": base_layout["plate_pile_idx"].tolist(),
        "onion_pile_idx": base_layout["onion_pile_idx"].tolist(),
        "tomato_pile_idx": base_layout["tomato_pile_idx"].tolist(),
        "pot_idx": base_layout["pot_idx"].tolist()
    }
    item_counts = {key: len(positions) for key, positions in items.items()}

    def generate_single_variant():
        """Generates a single variant layout ensuring items are adjacent to a floor space."""
        new_layout = dict(base_layout)  # Start with a copy of the base layout

        # Generate unique positions for each object
        new_positions = {}
        remaining_positions = set(available_positions)  # Keep track of remaining positions

        def is_adjacent_to_floor(pos):
            """Check if object position is reachable by floor tile."""
            width = base_layout["width"]
            height = base_layout["height"]

            row, col = pos // width, pos % width
            adjacent_positions = [
                (row, col - 1),  # Left
                (row, col + 1),  # Right
                (row - 1, col),  # Up
                (row + 1, col)   # Down
            ]

            # Check if any adjacent position is floor
            for r, c in adjacent_positions:
                if 0 <= r < height and 0 <= c < width: 
                    adj_pos = r * width + c
                    if adj_pos not in wall_positions:  
                        return True
            return False

        for item_type, count in item_counts.items():
            # Keep generating positions until valid ones are found
            valid_positions_found = False
            
            while not valid_positions_found:
                candidate_positions = random.sample(remaining_positions, count)
                if all(is_adjacent_to_floor(pos) for pos in candidate_positions):
                    chosen_positions = candidate_positions
                    valid_positions_found = True  # Valid positions found

            new_positions[item_type] = chosen_positions
            remaining_positions -= set(chosen_positions) 

        # Update new layout
        for item_type, positions in new_positions.items():
            new_layout[item_type] = jnp.array(positions)

        return FrozenDict(new_layout)
    
    for _ in range(max_variants):
        variant = generate_single_variant()
        variants.append(variant)

    return variants

overcooked_v2_layouts = {
    "cramped_room" : FrozenDict(cramped_room),
    "asymm_advantages" : FrozenDict(asymm_advantages),
    "coord_ring" : FrozenDict(coord_ring),
    "bottleneck_room" : layout_grid_to_dict(bottleneck_room_grid),
    "forced_coord" : FrozenDict(forced_coord),
    "counter_circuit" : layout_grid_to_dict(counter_circuit_grid),
    "cramped_room_tomatoes" : FrozenDict(cramped_room_tomatoes),    
    "asymm_advantages_tomatoes" : FrozenDict(asymm_advantages_tomatoes),     
    "big_room" : layout_grid_to_dict(big_room),     
    "L_shaped" : layout_grid_to_dict(L_shaped),     
    "U_shaped" : layout_grid_to_dict(U_shaped),     
    "gallery" : layout_grid_to_dict(gallery),     
    "single_wall" : layout_grid_to_dict(single_wall),     
    "too_many_cooks" : layout_grid_to_dict(too_many_cooks),     
    "signaling_room" : layout_grid_to_dict(signaling_room),     
    "H_shape" : layout_grid_to_dict(H_shape),     
    "two_islands" : layout_grid_to_dict(two_islands),     
    "semi_forced_coord" : layout_grid_to_dict(semi_forced_coord),     
}

base_layouts = overcooked_v2_layouts.copy()

for name, layout in base_layouts.items():
    variants = generate_variants(layout)
    for i, variant in enumerate(variants):
        variant_name = f"{name}_variant_{i}"
        overcooked_v2_layouts[variant_name] = variant