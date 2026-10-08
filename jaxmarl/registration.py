"""Register the CoordinationGrid task used by this project."""
from .environments.coordination_grid import CoordinationGrid

registered_envs = ["coordination_grid"]


def make(env_id: str, **env_kwargs):
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered jaxmarl environments.")
    return CoordinationGrid(**env_kwargs)
