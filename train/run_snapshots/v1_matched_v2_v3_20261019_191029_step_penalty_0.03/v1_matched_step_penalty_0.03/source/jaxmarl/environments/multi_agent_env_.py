""" 
Abstract base class for multi agent gym environments with JAX
Based on the Gymnax and PettingZoo APIs

"""

import jax
import jax.numpy as jnp
from typing import Dict
import chex
from functools import partial
from flax import struct
from typing import Tuple, Optional


@struct.dataclass
class State:
    done: chex.Array
    step: int


class MultiAgentEnv(object):
    """Jittable abstract base class for all jaxmarl Environments."""

    def __init__(
        self,
        num_agents: int,
    ) -> None:
        """
        num_agents (int): maximum number of agents within the environment, used to set array dimensions
        """
        self.num_agents = num_agents
        self.observation_spaces = dict()
        self.action_spaces = dict()

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Performs resetting of the environment."""
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
        reset_state: Optional[State] = None,
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Performs step transitions in the environment. Resets the environment if done.
        To control the reset state, pass `reset_state`. Otherwise, the environment will reset randomly."""

        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(key, state, actions)

        if reset_state is None:
            obs_re, states_re = self.reset(key_reset)
        else:
            states_re = reset_state
            obs_re = self.get_obs(states_re)

        # Auto-reset environment based on termination
        states = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st
        )
        obs = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )

        def _get_high_level_index(pos, land_pos, radius=0.2):
            """
            Returns integer index of nearest landmark if within `radius`,
            else returns -1.
            """
            dists = jnp.linalg.norm(land_pos - pos, axis=1)  # shape (num_landmarks,)
            min_idx = jnp.argmin(dists)
            min_dist = dists[min_idx]

            def yes_lm(_):
                return min_idx  # chosen landmark index
            def no_lm(_):
                return -1       # no landmark chosen

            return jax.lax.cond(min_dist < radius, yes_lm, no_lm, operand=None)


        def _hl_obs(aidx: int):
            """
            Return a single integer: which landmark the agent chooses.
            -1 if out of range
            """
            return _get_high_level_index(
                state.p_pos[aidx],      # agent position
                state.p_pos[self.num_agents:],  # all landmark positions
                radius=0.2
            )

        # def _hl_obs(aidx: int):
        #     # We only care about agent position and all landmark positions
        #     hl = self._get_high_level_action(
        #         state.p_pos[aidx],
        #         state.p_pos[self.num_agents:],  # the landmark positions
        #         radius=0.2
        #     )
        #     return hl

        # hl = {a: _hl_obs(i) for i, a in enumerate(self.agents)}
        # hl_actions = jnp.stack([_hl_obs(i) for i in range(self.num_agents)])  # (num_agents, num_landmarks)
        # hl_flat = hl_actions.reshape(-1)  # shape: (num_agents * num_landmarks,)
        # infos["high_level_actions"] = hl_flat
        hl_actions = jnp.array([_hl_obs(i) for i in range(self.num_agents)])  # shape (num_agents,)
        infos["high_level_actions"] = hl_actions

        return obs, states, rewards, dones, infos



    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""
        raise NotImplementedError

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Applies observation function to state."""
        raise NotImplementedError

    def observation_space(self, agent: str):
        """Observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """Action space for a given agent."""
        return self.action_spaces[agent]

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: State) -> Dict[str, chex.Array]:
        """Returns the available actions for each agent."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    def agent_classes(self) -> dict:
        """Returns a dictionary with agent classes, used in environments with hetrogenous agents.

        Format:
            agent_base_name: [agent_base_name_1, agent_base_name_2, ...]
        """
        raise NotImplementedError
