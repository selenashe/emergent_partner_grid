import collections

###############################################################################
# 1) Storing your skeleton layouts
###############################################################################
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
def generate_side_swapped_layouts():
    import copy

    # Which side partitions do we want to test?
    # e.g. we’ll test [("LR", partition_left_right), ("TB", partition_top_bottom)]
    # so we attempt both left↔right swaps and top↔bottom swaps.
    side_partitions = [
        ("LeftRight", partition_left_right),
        ("TopBottom", partition_top_bottom)
    ]

    # Which item‐swaps do we want to attempt across sides?
    # i.e. if side1 has O, side2 has B, and their counts match, we do an O↔B swap.
    # We can do a list, and do them one after another (like O↔B, O↔X, X↔P, etc.).
    # If you only want to do one swap at a time, adapt as needed.
    swaps_to_try = [
        ('O','B'),  # onion ↔ bowl
        ('O','X'),  # onion ↔ serving
        ('X','P'),  # serving ↔ pot
        # etc. – you can add more combos if needed
    ]

    results = []
    for layout_name, layout_str in layouts.items():
        base_grid = parse_layout(layout_str)

        for partition_name, partition_fn in side_partitions:
            side1, side2 = partition_fn(base_grid)
            
            # Now we want to consider *all* possible ways to do side‐based swaps.
            # For each combination, we clone the original, attempt the swaps,
            # then check BFS accessibility.
            # If you want to try *combinations* of all swaps (like O↔B then O↔X),
            # you can nest loops or do permutations. Below is a simple approach.
            # We'll do each swap in isolation so you see the pattern.
            for (chA, chB) in swaps_to_try:
                grid_copy = copy.deepcopy(base_grid)

                # Attempt the side‐swap for (chA <-> chB)
                swap_sides_if_same_count(grid_copy, side1, side2, chA, chB)

                # BFS check
                if all_items_accessible(grid_copy, agent_char='A', items=('B','O','X','P')):
                    # Convert to string
                    layout_after_swap = "\n".join("".join(row) for row in grid_copy)
                    info = (layout_name, partition_name, f"{chA}<->{chB}", layout_after_swap)
                    results.append(info)

    return results


if __name__ == "__main__":
    swapped_results = generate_side_swapped_layouts()
    for (layout_name, partition_name, swap_desc, layout_str) in swapped_results:
        print(f"\n=== {layout_name} | {partition_name} swap {swap_desc} ===")
        print(layout_str)

import pdb; pdb.set_trace()