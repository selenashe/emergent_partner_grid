"""Serialize and restore CoordinationGrid policy parameter trees."""
import os
from typing import Dict, Union
from safetensors.flax import save_file, load_file
from flax.traverse_util import flatten_dict, unflatten_dict

def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
    # Audit guide:
    # Flatten the JAX/Flax parameter tree and serialize its arrays in safetensors
    # format. This stores learned weights, not environment configuration or training
    # trajectories. The trainer saves resolved JSON configuration separately.
    #
    flattened_dict = flatten_dict(params, sep=',')
    save_file(flattened_dict, filename)

def load_params(filename:Union[str, os.PathLike]) -> Dict:
    # Audit guide:
    # Load saved tensor arrays and reconstruct their original nested parameter tree. The
    # evaluator must instantiate the same network architecture from the companion
    # configuration before applying these weights.
    #
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")
