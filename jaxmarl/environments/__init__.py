"""Shared environment interface and the project's CoordinationGrid task."""
from .multi_agent_env import MultiAgentEnv, State
from .coordination_grid import CoordinationGrid

__all__ = ["MultiAgentEnv", "State", "CoordinationGrid"]
