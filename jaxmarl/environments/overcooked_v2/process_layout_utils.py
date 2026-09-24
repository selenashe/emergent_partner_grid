import wandb
import numpy as np
import json 
import ast
import re
from pathlib import Path
import re


def clean_layout_string(layout_str):
    """Clean up the layout string to make it compatible with ast.literal_eval."""
    
    layout_str = re.sub(r',\s*dtype=\w+', '', layout_str)
    layout_str = re.sub(r'Array\((\[[^\]]*?\])\)', r'\1', layout_str)
    layout_str = re.sub(r'^FrozenDict\(\{', '{', layout_str)
    layout_str = re.sub(r'\}\)$', '}', layout_str)
    layout_str = re.sub(r'\b(int32|float32|int|float)\b', '', layout_str)
    layout_str = layout_str.replace('\n', '').replace('\t', ' ')
    layout_str = re.sub(r'\s+', ' ', layout_str).strip()
    layout_str = re.sub(r'\s*:\s*', ': ', layout_str)
    layout_str = re.sub(r'(\b[a-zA-Z_][\w]*\b)\s*:', r'"\1":', layout_str)
    layout_str = re.sub(r',\s*([\}\]])', r'\1', layout_str)
    layout_str = re.sub(r',\s*\}\)$', '}', layout_str)
    layout_str = re.sub(r',\s*\}$', '}', layout_str)
    layout_str = re.sub(r'\}\)$', '}', layout_str)

    if layout_str.count('{') != layout_str.count('}') or layout_str.count('[') != layout_str.count(']'):
        raise ValueError("Braces or brackets are mismatched in cleaned layout string.")

    return layout_str.strip()


def parse_layout(layout_str):
    """Parse the cleaned layout string into a Python dictionary."""
    layout_str = clean_layout_string(layout_str)
    try:
        parsed_layout = ast.literal_eval(layout_str)
    except Exception as e:
        raise ValueError(f"Failed to parse layout string after cleaning: {e}")
    return parsed_layout


def convert_to_grid(parsed_layout):
    """Convert layout indices to a grid representation."""
    grid = np.full((parsed_layout['height'], parsed_layout['width']), ' ', dtype=str)
    
    def get_row_col(idx):
        row, col = divmod(idx, parsed_layout['width'])
        return row, col
    
    # Fill in the walls
    for idx in parsed_layout['wall_idx']:
        row, col = get_row_col(idx)
        grid[row, col] = 'X'
    
    # Place agents
    agent_idx = 0
    for idx in parsed_layout['agent_idx']:
        agent_idx += 1
        row, col = get_row_col(idx)
        grid[row, col] = f'{agent_idx}'
    
    # Place goals
    for idx in parsed_layout['goal_idx']:
        row, col = get_row_col(idx)
        grid[row, col] = 'S'
    
    # Place plate piles
    for idx in parsed_layout['plate_pile_idx']:
        row, col = get_row_col(idx)
        grid[row, col] = 'D'
    
    # Place onion piles
    for idx in parsed_layout['onion_pile_idx']:
        row, col = get_row_col(idx)
        grid[row, col] = 'O'
    
    # Place tomato piles
    for idx in parsed_layout['tomato_pile_idx']:
        row, col = get_row_col(idx)
        grid[row, col] = 'T'
    
    for idx in parsed_layout['pot_idx']:
        row, col = get_row_col(idx)
        grid[row, col] = 'P'

    return '\n'.join([''.join(row) for row in grid])

def convert_layout_to_json(layout_str, output_file=None):
    """Convert layout string to desired JSON format."""
    parsed_layout = parse_layout(layout_str)
    grid = convert_to_grid(parsed_layout)
    json_data = {
        "grid": grid,
        "start_bonus_orders": [],
        "start_all_orders": [{"ingredients": ["onion"]}],
        "order_bonus": 2,
        "onion_value": 2,
        "tomato_value": 1,
        "onion_time": 10,
        "tomato_time": 10,
    }
    
    return json_data


grid_encoding = {
    'X': 1,  # Wall
    'S': 2,  # Goal
    'P': 3,  # Pot
    'D': 4,  # Plate pile
    'O': 5,  # Onion pile
    'T': 6,  # Tomato pile
    '1': 0,  # Agent 1
    '2': 0,  # Agent 2
    ' ': 0   # Empty space
}

def convert_layout_str_to_grid(layout_str):
    parsed_layout = parse_layout(layout_str)
    layout_grid = convert_to_grid(parsed_layout)
    grid_rows = layout_grid.split("\n")
    layout_grid = [[grid_encoding[cell] for cell in row] for row in grid_rows]
    return layout_grid