import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
import random
import json
import os
from itertools import product

random.seed(42)

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

counter_circuit_onion_grid = """
WWWPBPWWW
W   A   W
W WWWWW W
W   A   W
WWWOXOWWW
"""

bottleneck_room_grid = """
WBWWWTW
WA W  W
P   A W
X W  OW
WPWWWWW
"""


new_layout = """
WOXWWWOXW
WA PWWA P
B  OWB  O
WWWWWWWWW
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
WXPBWWW
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
POWBXTWWW
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
                # Empty 
                continue

    for key in symbol_to_key.values():
        layout_dict[key] = jnp.array(layout_dict[key])

    return FrozenDict(layout_dict)

def generate_variants(base_layout, obj_ranges=None, max_per_combination=3):
    """
    Generates layout variants by explicitly iterating over all combinations of item counts.
    """
    
    if obj_ranges is None:
        obj_ranges = {
            "pot_idx": (1, 2),
            "onion_pile_idx": (1, 2),
            "tomato_pile_idx": (0, 1),  
            "plate_pile_idx": (1, 2),
            "goal_idx": (1, 2),
        }

    variants = []
    wall_positions = base_layout["wall_idx"].tolist()
    agent_positions = set(base_layout["agent_idx"].tolist())
    available_positions = list(set(wall_positions) - agent_positions)  

    def generate_single_variant(combination):
        """Generates a single variant layout for a specific combination."""
        new_layout = dict(base_layout)  
        new_positions = {}
        remaining_positions = set(available_positions)  

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

            for r, c in adjacent_positions:
                if 0 <= r < height and 0 <= c < width:
                    adj_pos = r * width + c
                    if adj_pos not in wall_positions:
                        return True
            return False

        for item_type, count in zip(obj_ranges.keys(), combination):
            valid_positions_found = False
            while not valid_positions_found:
                candidate_positions = random.sample(remaining_positions, count)
                if all(is_adjacent_to_floor(pos) for pos in candidate_positions):
                    new_positions[item_type] = candidate_positions
                    remaining_positions -= set(candidate_positions)
                    valid_positions_found = True

        # Assign positions and recipe
        for item_type, positions in new_positions.items():
            new_layout[item_type] = jnp.array(positions)

        return FrozenDict(new_layout)

    # Generate all combinations
    count_ranges = [range(obj_ranges[key][0], obj_ranges[key][1] + 1) for key in obj_ranges.keys()]
    all_combinations = list(product(*count_ranges))

    for combination in all_combinations:
        for _ in range(max_per_combination):  
            variant = generate_single_variant(combination)
            variants.append(variant)

    return variants

overcooked_v2_layouts = {
    "cramped_room" : FrozenDict(cramped_room),
    "bottleneck_room" : layout_grid_to_dict(bottleneck_room_grid),
    "asymm_advantages" : FrozenDict(asymm_advantages),
    "coord_ring" : FrozenDict(coord_ring),
    # "forced_coord" : FrozenDict(forced_coord),
    "counter_circuit" : layout_grid_to_dict(counter_circuit_grid),
    # "counter_circuit_onion" : layout_grid_to_dict(counter_circuit_onion_grid),
    # "cramped_room_tomatoes" : FrozenDict(cramped_room_tomatoes),    
    # "asymm_advantages_tomatoes" : FrozenDict(asymm_advantages_tomatoes),     
    # "new_layout" : layout_grid_to_dict(new_layout),     
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

obj_ranges = {
    "pot_idx": (1, 2),
    "onion_pile_idx": (1, 2),
    "tomato_pile_idx": (0, 1),
    "plate_pile_idx": (1, 2),
    "goal_idx": (1, 2),
}


# for name, layout in base_layouts.items():  
#     variants = generate_variants(layout, obj_ranges=obj_ranges, max_per_combination=3)
#     for i, variant in enumerate(variants):
#         counts = {
#             key: len(variant[key]) if key in variant else 0
#             for key in ["pot_idx", "onion_pile_idx", "tomato_pile_idx", "plate_pile_idx", "goal_idx"]
#         }
#         count_str = f"{counts['pot_idx']}pots_{counts['onion_pile_idx']}onions_{counts['tomato_pile_idx']}tomatoes_{counts['goal_idx']}serving"
#         variant_name = f"{name}_variant_{i}_{count_str}"
#         overcooked_v2_layouts[variant_name] = variant


for name, layout in base_layouts.items():  
    variants = generate_variants(layout, obj_ranges=obj_ranges, max_per_combination=3)
    for i, variant in enumerate(variants):
        counts = {
            key: len(variant[key]) if key in variant else 0
            for key in ["pot_idx", "onion_pile_idx", "tomato_pile_idx", "plate_pile_idx", "goal_idx"]
        }
        count_str = f"{counts['pot_idx']}pots_{counts['onion_pile_idx']}onions_{counts['tomato_pile_idx']}tomatoes_{counts['goal_idx']}serving_{counts['plate_pile_idx']}bowls"
        variant_name = f"{name}_variant_{i}_{count_str}"
        overcooked_v2_layouts[variant_name] = variant


# layouts_to_dump = list(set(overcooked_v2_layouts.keys()))

# script_directory = os.path.dirname(os.path.abspath(__file__))
# output_file = os.path.join(script_directory, "overcooked_v2_layouts.json")

# with open(output_file, "w") as json_file:
#     json.dump(layouts_to_dump, json_file, indent=4)

# print(f"Saved layouts to {output_file}")

# Final combined layouts
# print("Generated layouts:", list(overcooked_v2_layouts.keys()))