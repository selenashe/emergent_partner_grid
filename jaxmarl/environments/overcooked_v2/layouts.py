import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
import random
import json
import os

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
    "onion_pile_idx" : jnp.array([15]),
    "plate_pile_idx" : jnp.array([5]),
    "tomato_pile_idx": jnp.array([10]),    
    "pot_idx" : jnp.array([9]),
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

#########################################################
# Experiment 1 Layouts
# cramped_room
# coord_ring_v2
# fivebyfive_v1
# cramped_room_v2
# semi_forced_coord_v1

# cramped_room_v1 = """
# WWPWW
# O   O
# W   W
# WBWXW
# """


# cramped_room
# coord_ring_v2
# fivebyfive_v1
# cramped_room_v3
# cramped_room_v4

coord_ring_v2 = """
WWWPW
W A W
W W W
B  AW
WOXWW
"""

fivebyfive_v1 = """
WWWPW
W W W
B   O
WA AW
WXWOW
"""


cramped_room_v3 = """
WWWWW
P   O
W   W
BA AW
WWXWW
"""

cramped_room_v4 = """
WWBWW
W   O
WA AW
WXWPW
"""
########################


cramped_room_v2 = """
WPWWW
W   O
BA AW
WWXWW
"""

semi_forced_coord_v1 = """
WWWXW
W P W
W W W
O A B
WWAWW
WWWWW
"""

semi_forced_coord_v2 = """
WWWXW
W P W
O W W
W A B
W A W
WWWWW
"""

########################


coord_ring2_tomato = """
WWWPW
W A W
X W W
B  AW
WOTWW
"""

semi_forced_coord_v2 = """
WPWWW
X W W
W W B
O A W
WWAWW
WWWWW
"""

forced_coord2 = """
WWWWW
W W P
W X W
OABAO
WWWWW
"""

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

counter_circuit_onion_grid2 = """
WWWPBPWWW
W   A   W
W WWWWW W
W      AW
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

###################################################
# PARTIAL DIVIDER IN THE MIDDLE
partial_divider = """
WWXWBWWW
WA     W
WWPWW  O
WA     W
WWXWWWWW
"""

# TWO DIVIDERS
two_dividers = """
WWWBWWWW
WA     W
WWPWW  W
X      O
WWWWW  W
W A    W
WPWWWWWW
"""


# ISLANDS IN THE MIDDLE
islands_in_middle = """
WWXPWWW
WA    W
O  W  T
W     W
W  W  W
WA    W
WWBPXWW
"""

# UNUSUAL SHAPE
unusual_shape = """
WWWWWW
O A WW
W WWWW
PA   X
WWWWBW
"""

# CONE SHAPE
cone_shape = """
WWWWWWW
O A   
W A  
P   X
W  
W B
WW
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
                # Empty 
                continue

    for key in symbol_to_key.values():
        layout_dict[key] = jnp.array(layout_dict[key])

    return FrozenDict(layout_dict)

def generate_variants(base_layout, max_variants=100):
    variants = []

    wall_positions = base_layout["wall_idx"].tolist() 
    agent_positions = set(base_layout["agent_idx"].tolist())  
    available_positions = list(set(wall_positions) - agent_positions) 

    items = {
        "goal_idx": base_layout["goal_idx"].tolist(),
        "plate_pile_idx": base_layout["plate_pile_idx"].tolist(),
        "onion_pile_idx": base_layout["onion_pile_idx"].tolist(),
        "tomato_pile_idx": base_layout["tomato_pile_idx"].tolist(),
        "pot_idx": base_layout["pot_idx"].tolist(),
        "agent_idx": base_layout["agent_idx"].tolist()
    }
    item_counts = {key: len(positions) for key, positions in items.items()}

    def generate_single_variant():
        """Generates a single variant layout ensuring items are adjacent to a floor space."""
        new_layout = dict(base_layout)  
        width, height = base_layout["width"], base_layout["height"]
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

        for item_type, count in item_counts.items():
            attempts = 0
            max_attempts = 100  # Avoid infinite loops
            valid_positions_found = False

            while not valid_positions_found and attempts < max_attempts:
                # Ensure there are enough positions available
                if len(remaining_positions) < count:
                    raise ValueError(f"Not enough positions available for {item_type}. Required: {count}, Available: {len(remaining_positions)}")

                # Sample candidate positions
                candidate_positions = random.sample(remaining_positions, count)

                # Check adjacency to floor tiles
                if all(is_adjacent_to_floor(pos, width, height) for pos in candidate_positions):
                    new_positions[item_type] = candidate_positions
                    remaining_positions -= set(candidate_positions)
                    valid_positions_found = True

                attempts += 1

            if not valid_positions_found:
                raise RuntimeError(f"Failed to find valid positions for {item_type} after {max_attempts} attempts.")

        for item_type, positions in new_positions.items():
            new_layout[item_type] = jnp.array(positions)

        return FrozenDict(new_layout)
    
    for _ in range(max_variants):
        variant = generate_single_variant()
        variants.append(variant)

    return variants

def generate_random_layout(height, width):
    """Generate a random layout and make sure it's solvable."""
    layout = {
        "height": height,
        "width": width,
        "wall_idx": jnp.array([]),
        "agent_idx": jnp.array([]),
        "goal_idx": jnp.array([]),
        "plate_pile_idx": jnp.array([]),
        "onion_pile_idx": jnp.array([]),
        "tomato_pile_idx": jnp.array([]),
        "pot_idx": jnp.array([])
    }

    num_clusters = 3  
    cluster_size = int(0.1 * height * width)  
    wall_positions = generate_wall_clusters(height, width, num_clusters, cluster_size)
    layout["wall_idx"] = jnp.array(list(wall_positions))

    for key in ["goal_idx", "plate_pile_idx", "onion_pile_idx", "tomato_pile_idx", "pot_idx"]:
        layout[key] = jnp.array([random.choice(range(height * width))])

    layout = adjust_positions(layout)
    return FrozenDict(layout)

def neighbors(pos, height, width):
    row, col = divmod(pos, width)
    potential_neighbors = [
        (row - 1, col),  # Up
        (row + 1, col),  # Down
        (row, col - 1),  # Left
        (row, col + 1)   # Right
    ]
    valid_neighbors = [
        r * width + c for r, c in potential_neighbors
        if 0 <= r < height and 0 <= c < width
    ]
    return valid_neighbors

def is_solvable(layout):
    height, width = layout["height"], layout["width"]
    key_positions = (
        layout["goal_idx"].tolist() +
        layout["plate_pile_idx"].tolist() +
        layout["onion_pile_idx"].tolist() +
        layout["tomato_pile_idx"].tolist() +
        layout["pot_idx"].tolist() + 
        layout["agent_idx"].tolist()
    )
    
    if not key_positions:
        return False  

    walkable = set(range(height * width)) - set(layout["wall_idx"].tolist())
    key_positions = [pos for pos in key_positions if pos in walkable]
    if not key_positions:
        return False  

    adjacency = {
        pos: [n for n in neighbors(pos, height, width) if n in walkable]
        for pos in walkable
    }

    visited = set()
    stack = [key_positions[0]] 
    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            stack.extend(adjacency[current])

    return all(pos in visited for pos in key_positions)

def generate_wall_clusters(height, width, num_clusters, cluster_size):
    """
    Generate continuous wall clusters on a grid.
    """
    wall_positions = set()

    def is_within_bounds(pos):
        """Check if a position is within grid bounds."""
        row, col = divmod(pos, width)
        return 0 <= row < height and 0 <= col < width

    def neighbors(pos):
        """Get valid neighbors of a position in the grid."""
        row, col = divmod(pos, width)
        potential_neighbors = [
            (row - 1, col),  # Up
            (row + 1, col),  # Down
            (row, col - 1),  # Left
            (row, col + 1)   # Right
        ]
        return [
            r * width + c for r, c in potential_neighbors
            if 0 <= r < height and 0 <= c < width
        ]

    for _ in range(num_clusters):
        cluster_seed = random.randint(0, height * width - 1)
        cluster = {cluster_seed}  
        frontier = [cluster_seed]

        while len(cluster) < cluster_size and frontier:
            current = frontier.pop()
            for neighbor in neighbors(current):
                if neighbor not in cluster and is_within_bounds(neighbor) and random.random() < 0.7:
                    cluster.add(neighbor)
                    frontier.append(neighbor)

        wall_positions.update(cluster)

    return wall_positions

def adjust_positions(layout):
    """Repair invalid layouts by repositioning key objects."""
    wall_positions = set(layout["wall_idx"].tolist())
    walkable_positions = list(set(range(layout["height"] * layout["width"])) - wall_positions)

    for key in ["goal_idx", "plate_pile_idx", "onion_pile_idx", "tomato_pile_idx", "pot_idx", "agent_idx"]:
        if not layout[key].size:  
            layout[key] = jnp.array([random.choice(walkable_positions)])
    return layout

def generate_new_base_layouts(num_new_layouts, height, width):
    new_layouts = {}
    for i in range(num_new_layouts):
        layout = generate_random_layout(height, width)
        if is_solvable(layout):
            new_layouts[f"generated_base_{i}"] = layout
    return new_layouts

def generate_variants(base_layout, max_variants=10, obj_ranges=None):
    """
    Generates layout variants by choosing unique positions for items with varying counts.
    """
    if obj_ranges is None:
        obj_ranges = {
            "goal_idx": (1, 1), 
            "plate_pile_idx": (1, 2),
            "onion_pile_idx": (1, 3),
            "tomato_pile_idx": (1, 3),
            "pot_idx": (1, 2),
        }

    variants = []

    wall_positions = base_layout["wall_idx"].tolist()
    agent_positions = set(base_layout["agent_idx"].tolist())
    available_positions = list(set(wall_positions) - agent_positions)  

    def generate_single_variant():
        """Generates a single variant layout ensuring items are adjacent to a floor space."""
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

        for item_type, (min_count, max_count) in obj_ranges.items():
            count = random.randint(min_count, max_count)  
            valid_positions_found = False

            while not valid_positions_found:
                candidate_positions = random.sample(remaining_positions, count)
                if all(is_adjacent_to_floor(pos) for pos in candidate_positions):
                    new_positions[item_type] = candidate_positions
                    remaining_positions -= set(candidate_positions)
                    valid_positions_found = True

        for item_type, positions in new_positions.items():
            new_layout[item_type] = jnp.array(positions)

        return FrozenDict(new_layout)

    for _ in range(max_variants):
        variant = generate_single_variant()
        variants.append(variant)

    return variants

overcooked_v2_layouts = {
    "cramped_room" : FrozenDict(cramped_room),
    "cramped_room_v2" : layout_grid_to_dict(cramped_room_v2),
    "cramped_room_v3" : layout_grid_to_dict(cramped_room_v3),
    "cramped_room_v4" : layout_grid_to_dict(cramped_room_v4),
    "fivebyfive_v1" : layout_grid_to_dict(fivebyfive_v1),
    "bottleneck_room" : layout_grid_to_dict(bottleneck_room_grid),
    "asymm_advantages" : FrozenDict(asymm_advantages),
    "coord_ring" : FrozenDict(coord_ring),
    "coord_ring_v2" : layout_grid_to_dict(coord_ring_v2),
    "coord_ring2_tomato" : layout_grid_to_dict(coord_ring2_tomato),
    "forced_coord" : FrozenDict(forced_coord),
    "forced_coord2" : layout_grid_to_dict(forced_coord2),
    "semi_forced_coord_v1" : layout_grid_to_dict(semi_forced_coord_v1),
    "semi_forced_coord_v2" : layout_grid_to_dict(semi_forced_coord_v2),
    "counter_circuit" : layout_grid_to_dict(counter_circuit_grid),
    "counter_circuit_onion" : layout_grid_to_dict(counter_circuit_onion_grid),
    "counter_circuit_onion2" : layout_grid_to_dict(counter_circuit_onion_grid2),
    "cramped_room_tomatoes" : FrozenDict(cramped_room_tomatoes),    
    "asymm_advantages_tomatoes" : FrozenDict(asymm_advantages_tomatoes),     
    "new_layout" : layout_grid_to_dict(new_layout),     
    "big_room" : layout_grid_to_dict(big_room),     
    "partial_divider" : layout_grid_to_dict(partial_divider),     
    "two_dividers" : layout_grid_to_dict(two_dividers),     
    "islands_in_middle" : layout_grid_to_dict(islands_in_middle),     
    "unusual_shape" : layout_grid_to_dict(unusual_shape),     
    "cone_shape" : layout_grid_to_dict(cone_shape),        
}

# new_base_layouts = generate_new_base_layouts(num_new_layouts=5, height=6, width=6)
# overcooked_v2_layouts.update(new_base_layouts)
base_layouts = overcooked_v2_layouts.copy()

obj_ranges = {
    "pot_idx": (1, 2),
    "onion_pile_idx": (1, 2),
    "tomato_pile_idx": (1, 2),
    "plate_pile_idx": (1, 2),
    "goal_idx": (1, 1),
}

for name, layout in base_layouts.items():  
    variants = generate_variants(layout, max_variants=100, obj_ranges=obj_ranges)
    for i, variant in enumerate(variants):
        counts = {
            key: len(variant[key]) for key in ["pot_idx", "onion_pile_idx", "tomato_pile_idx"]
        }
        count_str = f"pots{counts['pot_idx']}_onions{counts['onion_pile_idx']}_tomatoes{counts['tomato_pile_idx']}"
        variant_name = f"{name}_variant_{i}_{count_str}"
        overcooked_v2_layouts[variant_name] = variant

# Final combined layouts
# print("Generated layouts:", overcooked_v2_layouts)