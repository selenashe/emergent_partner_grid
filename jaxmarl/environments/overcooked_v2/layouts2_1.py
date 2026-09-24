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


layouts = {
    "fivebyfive_1pots_2onion_1serving_1plates_open": """
WWPWW
W   W
B   X
WA AW
WOWOW
""",
    "fivebyfive_2pots_2onion_1serving_1plates_open": """
WWPPW
W   W
B   X
WA AW
WOWOW
""",
    "fivebyfive_1pots_2onion_1serving_1plates_closed": """
WWPWW
WW WW
B   X
WA AW
WOWOW
""",
    "fivebyfive_2pots_2onion_1serving_1plates_closed": """
WWPWW
WW PW
B   X
WA AW
WOWOW
""",
    "fivebyfive_2pots_2onion_1serving_1plates_divider": """
WPWPW
W W W
B   X
WA AW
WOWOW
""",
    "fivebyfive_1pots_2onion_1serving_1plates_divider": """
WWWPW
W W W
B   X
WA AW
WOWOW
""",
    "fivebyfive_1pots_2onion_1serving_1plates_island": """
WWWPW
B   W
W W X
WA AW
WOWOW
""",
    "fivebyfive_2pots_2onion_1serving_1plates_island": """
WWPPW
B   W
W W X
WA AW
WOWOW
"""
}

###############################################################################
# 2) Convert layout string to 2D grid
###############################################################################
def parse_layout(layout_str):
    lines = [line for line in layout_str.strip('\n').split('\n')]
    grid = [list(row) for row in lines]
    return grid

###############################################################################
# 3) Identify sides
###############################################################################
def partition_left_right(grid):
    R, C = len(grid), len(grid[0])
    left_cells = []
    right_cells = []
    for r in range(R):
        for c in range(C):
            if c < (C // 2):
                left_cells.append((r, c))
            elif c > (C // 2):
                right_cells.append((r, c))
    return left_cells, right_cells

def partition_top_bottom(grid):
    R, C = len(grid), len(grid[0])
    top_cells = []
    bottom_cells = []
    for r in range(R):
        for c in range(C):
            if r < (R // 2):
                top_cells.append((r, c))
            elif r > (R // 2):
                bottom_cells.append((r, c))
    return top_cells, bottom_cells

###############################################################################
# 4) Count items on each side, compare, and swap if counts match
###############################################################################
def swap_sides_if_same_count(grid, side1_cells, side2_cells, char1, char2):
    side1_char1_positions = [(r, c) for (r, c) in side1_cells if grid[r][c] == char1]
    side2_char2_positions = [(r, c) for (r, c) in side2_cells if grid[r][c] == char2]
    if len(side1_char1_positions) == len(side2_char2_positions):
        for (r, c) in side1_char1_positions:
            grid[r][c] = char2
        for (r, c) in side2_char2_positions:
            grid[r][c] = char1

###############################################################################
# 5) BFS Accessibility: Check that every special item is reachable from 'A'
###############################################################################
def find_positions(grid, chars):
    found = {ch: [] for ch in chars}
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] in chars:
                found[grid[r][c]].append((r, c))
    return found

def is_accessible(grid, start_pos, end_pos):
    if start_pos == end_pos:
        return True
    R, C = len(grid), len(grid[0])
    queue = collections.deque([start_pos])
    visited = set([start_pos])
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        r, c = queue.popleft()
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C:
                if grid[nr][nc] != 'W' and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
                    if (nr, nc) == end_pos:
                        return True
    return False

def all_items_accessible(grid, agent_char='A', items=('B', 'O', 'X', 'P')):
    positions = find_positions(grid, [agent_char] + list(items))
    if not positions[agent_char]:
        return False
    agent_pos = positions[agent_char][0]
    for item_char in items:
        for pos in positions[item_char]:
            if not is_accessible(grid, agent_pos, pos):
                return False
    return True

###############################################################################
# 6) Convert grid layout to dictionary
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

###############################################################################
# 7) Generate random workstation layouts
###############################################################################
def generate_random_workstation_layouts():
    import copy
    workstation_chars = ['O', 'P', 'X', 'B']
    results = {}
    for layout_name, layout_str in layouts.items():
        base_grid = parse_layout(layout_str)
        positions = []
        items = []
        for r in range(len(base_grid)):
            for c in range(len(base_grid[0])):
                if base_grid[r][c] in workstation_chars:
                    positions.append((r, c))
                    items.append(base_grid[r][c])
        unique_configs = set()
        config_count = 0
        attempts = 0
        while config_count < 10 and attempts < 100:
            attempts += 1
            shuffled_items = items.copy()
            random.shuffle(shuffled_items)
            config_key = tuple(shuffled_items)
            if config_key in unique_configs:
                continue
            unique_configs.add(config_key)
            config_count += 1
            new_grid = copy.deepcopy(base_grid)
            for idx, (r, c) in enumerate(positions):
                new_grid[r][c] = shuffled_items[idx]
            layout_dict = layout_grid_to_dict("\n".join("".join(row) for row in new_grid))
            config_name = f"{layout_name}_shuffle_{config_count}"
            results[config_name] = layout_dict
    return results


swapped_results = generate_random_workstation_layouts()
swapped_results['cramped_room'] = FrozenDict(cramped_room)
swapped_results['counter_circuit_v2'] = layout_grid_to_dict(counter_circuit_v2)
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
