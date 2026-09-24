import collections
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from itertools import product


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


counter_circuit_v1 = """
WXBWWBXW
W      W
P WWWW P
WA    AW
WTOWWOTW
"""


counter_circuit_v2 = """
WBPXW
WA  W
WWWWW
WA  W
WOTWW
"""

counter_circuit_v2 = """
WWWWPWWW
W     AT
W WWWW O
XA     W
WBWWWWWW
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
    "fivebyfive_1pots_2onion_1serving_1plates_verticaldivider": """
WWWPW
B   W
WWW X
WA AW
WOWOW
""",
    "fivebyfive_2pots_2onion_1serving_1plates_verticaldivider": """
WWPPW
B   W
WWW X
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
#    In this example, we define "left side" vs. "right side," and also
#    "top side" vs. "bottom side." You can adapt how you define “sides”
#    for your own scenario.
###############################################################################
def partition_left_right(grid):
    """
    Partition cells into left side vs right side.
    For a 5-wide grid, columns 0,1 are 'left', columns 3,4 are 'right'
    (middle column 2 can be ignored or assigned arbitrarily).
    Returns (left_cells, right_cells) as lists of (r, c) coords.
    """
    R, C = len(grid), len(grid[0])
    left_cells = []
    right_cells = []

    # For a 5x5, you might define columns [0..1] as left, [3..4] as right,
    # ignoring column 2. Or do something else—up to you.
    for r in range(R):
        for c in range(C):
            # skip walls if you prefer, or just collect them anyway
            if c < (C // 2):
                left_cells.append((r, c))
            elif c > (C // 2):
                right_cells.append((r, c))
            # You can decide what to do if c == C//2 is the middle column
            # e.g. ignore, or treat it all as left or right
    return left_cells, right_cells


def partition_top_bottom(grid):
    """
    Partition cells into top side vs bottom side.
    For a 5-high grid, rows 0,1 are 'top', rows 3,4 are 'bottom',
    ignoring row 2. Adjust as you see fit.
    """
    R, C = len(grid), len(grid[0])
    top_cells = []
    bottom_cells = []

    for r in range(R):
        for c in range(C):
            if r < (R // 2):
                top_cells.append((r, c))
            elif r > (R // 2):
                bottom_cells.append((r, c))
            # row == R//2 is the middle row—handle as you prefer

    return top_cells, bottom_cells

###############################################################################
# 4) Count items on each side, compare, and swap if counts match
###############################################################################
def swap_sides_if_same_count(grid, side1_cells, side2_cells, char1, char2):
    """
    If side1 has the same number of `char1` as side2 has `char2`, swap them:
      - Every `char1` in side1 becomes `char2`,
      - Every `char2` in side2 becomes `char1`.
    Otherwise do nothing.
    NOTE: This modifies grid in place.
    """
    side1_char1_positions = [(r,c) for (r,c) in side1_cells if grid[r][c] == char1]
    side2_char2_positions = [(r,c) for (r,c) in side2_cells if grid[r][c] == char2]

    if len(side1_char1_positions) == len(side2_char2_positions):
        # Perform the actual swap
        for (r, c) in side1_char1_positions:
            grid[r][c] = char2
        for (r, c) in side2_char2_positions:
            grid[r][c] = char1

###############################################################################
# 5) BFS Accessibility: Check that every special item is reachable from 'A'
###############################################################################
def find_positions(grid, chars):
    """Return dict mapping each char -> list of (r,c) found."""
    found = {ch: [] for ch in chars}
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] in chars:
                found[grid[r][c]].append((r, c))
    return found

def is_accessible(grid, start_pos, end_pos):
    """Simple BFS ignoring walls 'W'."""
    if start_pos == end_pos:
        return True

    R, C = len(grid), len(grid[0])
    queue = collections.deque([start_pos])
    visited = set([start_pos])
    moves = [(-1,0), (1,0), (0,-1), (0,1)]

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

def all_items_accessible(grid, agent_char='A', items=('B','O','X','P')):
    """
    Check from the first 'A' found if all items in `items` are reachable.
    """
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
# 6) Main logic: for each layout, try side‐swaps, check BFS, store results
###############################################################################
# def generate_side_swapped_layouts():
#     import copy
#     side_partitions = [
#         ("LeftRight", partition_left_right),
#         ("TopBottom", partition_top_bottom)
#     ]

#     swaps_to_try = [
#         ('O','B'),  # onion ↔ bowl
#         ('O','X'),  # onion ↔ serving
#         ('X','P'),  # serving ↔ pot
#         # etc. – you can add more combos if needed
#     ]

#     results = []
#     for layout_name, layout_str in layouts.items():
#         base_grid = parse_layout(layout_str)

#         for partition_name, partition_fn in side_partitions:
#             side1, side2 = partition_fn(base_grid)
#             for (chA, chB) in swaps_to_try:
#                 grid_copy = copy.deepcopy(base_grid)

#                 swap_sides_if_same_count(grid_copy, side1, side2, chA, chB)

#                 # BFS check
#                 if all_items_accessible(grid_copy, agent_char='A', items=('B','O','X','P')):
#                     # Convert to string
#                     layout_after_swap = "\n".join("".join(row) for row in grid_copy)
#                     info = (layout_name, partition_name, f"{chA}<->{chB}", layout_after_swap)
#                     results.append(info)

#     return results

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



def generate_side_swapped_layouts():
    import copy

    # Which side partitions do we want to test?
    side_partitions = [
        ("LeftRight", partition_left_right),
        ("TopBottom", partition_top_bottom)
    ]

    # Which item-swaps do we want to attempt across sides?
    swaps_to_try = [
        ('O', 'B'),  # onion ↔ bowl
        ('O', 'X'),  # onion ↔ serving
        ('X', 'P'),  # serving ↔ pot
    ]

    results = {}
    for layout_name, layout_str in layouts.items():
        base_grid = parse_layout(layout_str)

        for partition_name, partition_fn in side_partitions:
            side1, side2 = partition_fn(base_grid)

            for (chA, chB) in swaps_to_try:
                grid_copy = copy.deepcopy(base_grid)

                # Attempt the side-swap for (chA <-> chB)
                swap_sides_if_same_count(grid_copy, side1, side2, chA, chB)

                # BFS check
                if all_items_accessible(grid_copy, agent_char='A', items=('B', 'O', 'X', 'P')):
                    # Convert to FrozenDict
                    layout_after_swap = layout_grid_to_dict("\n".join("".join(row) for row in grid_copy))
                    # Construct a unique name for this swapped layout
                    swapped_layout_name = f"{layout_name}_{partition_name}_{chA}_{chB}"
                    # Store in results
                    results[swapped_layout_name] = layout_after_swap

    return results



swapped_results = generate_side_swapped_layouts()
swapped_results['cramped_room'] = FrozenDict(cramped_room)
swapped_results['counter_circuit_v1'] = layout_grid_to_dict(counter_circuit_v1)
swapped_results['counter_circuit_v2'] = layout_grid_to_dict(counter_circuit_v2)

overcooked_v2_layouts = swapped_results

# if __name__ == "__main__":
#     for (layout_name, partition_name, swap_desc, layout_str) in swapped_results:
#         print(f"\n=== {layout_name} | {partition_name} swap {swap_desc} ===")
#         print(layout_str)