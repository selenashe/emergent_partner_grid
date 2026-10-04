"""Unified IPPO trainer for CoordinationGrid — the four experimental
conditions supported by config switches.

Switches:
    MODEL_TYPE      : "rnn" | "mlp"
    PARTNER_REGIME  : "diverse" | "single"
    INFLUENCE       : true | false

The four core conditions used in the replication:

    (rnn,  diverse, true)   -- main condition (multi-partner RNN)
    (mlp,  diverse, true)   -- memory control
    (rnn,  single,  true)   -- partner-diversity control
    (rnn,  diverse, false)  -- influence-pressure control

Only agent_0 is learned; the scripted partner runs inside the env. Obs
is a dict {"grid": (H, W, 5), "last_allocation": (3,), "is_t0": ()}:
the grid goes through a CNN, the allocation one-hot through a small
dense embedding, and both are concatenated before either a GRU (rnn)
or a plain MLP (mlp). Actor head is a flat 15-way Categorical over the
factored (move, alloc) ego action.

Each partner episode has a fixed 2-D capability vector (d_R, d_B); the
trainer materializes a long (capability_pair, layout_seq) schedule via
``episode_scheduler.build_schedule`` and advances a per-slot cursor.
"""

import functools
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence

# Make the containing train/ dir importable so we can pull
# episode_scheduler regardless of the CWD hydra ends up in.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper, save_params, load_params
from jaxmarl.environments.coordination_grid import (
    CoordinationGrid,
    COMM_ACTION_ONLY,
    TRAIN_CAPABILITY_PAIRS,
    TEST_CAPABILITY_PAIRS,
    DEFAULT_SINGLE_PARTNER,
    ego_action_mask,
    ALLOCATION_PROTOCOL,
)
from episode_scheduler import build_schedule, initial_episode_cursor, summarize_schedule


# ---------------------------------------------------------------------------
# Network modules
# ---------------------------------------------------------------------------

class ScannedRNN(nn.Module):
    """GRU cell that resets its hidden state on each ``done`` flag.

    Applied via ``nn.scan`` over the leading time axis.
    """

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, jnp.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )


class CNN(nn.Module):
    """The Overcooked-v2 baseline CNN, unchanged."""

    output_size: int = 64
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x):
        for feats, k in [(128, 1), (128, 1), (8, 1), (16, 3), (32, 3), (32, 3)]:
            x = nn.Conv(
                features=feats,
                kernel_size=(k, k),
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = self.activation(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(
            features=self.output_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        return x


class CommObsEncoder(nn.Module):
    """Encode the CoordinationGrid dict obs.

    grid: (N, H, W, 5) -> CNN -> (N, grid_dim)
    last_allocation: (N, 3) -> Dense -> (N, msg_dim)
    concatenate -> Dense(out_dim) -> (N, out_dim)

    The final projection to ``out_dim`` matters because the downstream
    ScannedRNN sizes its GRU cell from ``ins.shape[-1]`` and re-initializes
    its carry at that width. If encoder output width ≠ GRU carry width, the
    reset-on-done ``jnp.where`` blows up. So we lock encoder output width to
    ``out_dim`` (== GRU_HIDDEN_DIM at the call site).
    """

    out_dim: int = 128
    grid_dim: int = 64
    msg_dim: int = 8
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, obs):
        grid = obs["grid"].astype(jnp.float32)
        msg = obs["last_allocation"].astype(jnp.float32)

        grid_emb = CNN(
            output_size=self.grid_dim, activation=self.activation
        )(grid)

        msg_emb = nn.Dense(
            features=self.msg_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(msg)
        msg_emb = self.activation(msg_emb)

        joint = jnp.concatenate([grid_emb, msg_emb], axis=-1)
        out = nn.Dense(
            features=self.out_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(joint)
        out = self.activation(out)
        return out


class ActorCriticCommRNN(nn.Module):
    """Single-agent actor-critic with recurrent memory over dict obs."""

    action_dim: int
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x  # obs is a dict of leaves with leading (T, N, ...) shape
        activation = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh

        encoder = CommObsEncoder(
            out_dim=self.config["GRU_HIDDEN_DIM"],
            grid_dim=self.config.get("GRID_EMB_DIM", 64),
            msg_dim=self.config.get("MSG_EMB_DIM", 8),
            activation=activation,
        )
        # vmap encoder across the leading T dim; leaves are all keyed on the same
        # first axis, and jax.vmap on a pytree vmaps every leaf uniformly.
        embedding = jax.vmap(encoder)(obs)
        embedding = nn.LayerNorm()(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_mean)   # (T, N, action_dim=15)

        # t=0 is supplied by the environment, not a learned allocation.
        legal = ego_action_mask(obs)
        logits = jnp.where(legal, logits, jnp.full_like(logits, -jnp.inf))
        pi = distrax.Categorical(logits=logits)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class ActorCriticCommMLP(nn.Module):
    """Feed-forward MLP variant. Same call interface as the RNN model so
    the trainer can swap them via config: takes ``(hidden, (obs, dones))``
    and returns ``(hidden, pi, value)`` — but ``hidden`` is a dummy that
    is passed through unchanged. No cross-step memory.
    """

    action_dim: int
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x  # dones unused; kept for interface parity
        activation = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh

        encoder = CommObsEncoder(
            out_dim=self.config["GRU_HIDDEN_DIM"],
            grid_dim=self.config.get("GRID_EMB_DIM", 64),
            msg_dim=self.config.get("MSG_EMB_DIM", 8),
            activation=activation,
        )
        embedding = jax.vmap(encoder)(obs)
        embedding = nn.LayerNorm()(embedding)

        # One extra dense in place of the RNN cell so parameter counts
        # are broadly comparable.
        embedding = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        embedding = activation(embedding)

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_mean)

        legal = ego_action_mask(obs)
        logits = jnp.where(legal, logits, jnp.full_like(logits, -jnp.inf))
        pi = distrax.Categorical(logits=logits)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)


def build_network(config, action_dim):
    """Return the (network, initial-hstate factory) pair per MODEL_TYPE."""
    model_type = str(config.get("MODEL_TYPE", "rnn")).lower()
    if model_type == "rnn":
        network = ActorCriticCommRNN(action_dim=action_dim, config=config)

        def init_hstate(batch_size):
            return ScannedRNN.initialize_carry(
                batch_size, config["GRU_HIDDEN_DIM"]
            )
        return network, init_hstate, model_type
    if model_type == "mlp":
        network = ActorCriticCommMLP(action_dim=action_dim, config=config)

        def init_hstate(batch_size):
            # Dummy carry — must have leading batch dim so vmap/scan don't
            # complain, but the MLP model ignores its value.
            return jnp.zeros((batch_size, config["GRU_HIDDEN_DIM"]),
                             dtype=jnp.float32)
        return network, init_hstate, model_type
    raise ValueError(
        f"MODEL_TYPE must be 'rnn' or 'mlp'; got {model_type!r}"
    )


def resolve_training_capability_pool(config):
    """Return the (K, 2) list-of-lists of training capability pairs
    determined by PARTNER_REGIME.
    """
    regime = str(config.get("PARTNER_REGIME", "diverse")).lower()
    if regime == "diverse":
        return [list(p) for p in TRAIN_CAPABILITY_PAIRS]
    if regime == "single":
        single = tuple(config.get("SINGLE_PARTNER", DEFAULT_SINGLE_PARTNER))
        if len(single) != 2:
            raise ValueError(
                f"SINGLE_PARTNER must be a length-2 pair; got {single!r}"
            )
        return [[int(single[0]), int(single[1])]]
    raise ValueError(
        f"PARTNER_REGIME must be 'diverse' or 'single'; got {regime!r}"
    )


class Transition(NamedTuple):
    done: jnp.ndarray
    reset: jnp.ndarray  # pre-observation reset for GRU replay; done is post-step for GAE
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: Any  # dict pytree {"grid": ..., "last_allocation": ...}
    info: Dict
    # env-state fields captured for logging (BEFORE step_env ran on this step)
    pre_step_time: jnp.ndarray  # (N,) int32


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def require_current_protocol(config):
    if config.get("ALLOCATION_PROTOCOL") != ALLOCATION_PROTOCOL:
        raise ValueError(
            f"This implementation requires ALLOCATION_PROTOCOL={ALLOCATION_PROTOCOL!r}. "
            "Untagged/older configs and checkpoints belong to the fixed-allocation "
            "experiment; use its original code to evaluate them."
        )


def make_train(config):
    require_current_protocol(config)
    # Resolve capability pool from PARTNER_REGIME, then override the pool
    # in the env kwargs so the env sees exactly what we sampled from.
    env_kwargs = dict(config["ENV_KWARGS"])
    training_cap_pool = resolve_training_capability_pool(config)
    env_kwargs["partner_capability_pairs"] = training_cap_pool
    # Route INFLUENCE from top-level into the env (defaults true).
    if "INFLUENCE" in config:
        env_kwargs["influence"] = bool(config["INFLUENCE"])
    config["ENV_KWARGS"] = env_kwargs

    env = jaxmarl.make(config["ENV_NAME"], **env_kwargs)

    # Only action_only is supported in this build.
    if env.communication_condition != COMM_ACTION_ONLY:
        raise ValueError(
            f"only communication_condition='action_only' is supported, "
            f"got {env.communication_condition!r}"
        )

    # ONE learned actor per env (partner is scripted inside the env).
    config["NUM_ACTORS"] = config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # NOTE: LogWrapper is intentionally NOT used here. The final experiment
    # needs a scheduled auto-reset (each partner-episode consumes the next
    # pre-planned (capability, layout_seq) from the scheduler), which the
    # trainer implements manually inside _env_step below. LogWrapper's
    # episode-return bookkeeping is reimplemented in the trainer directly.

    def create_learning_rate_fn():
        base_lr = config["LR"]
        update_steps = config["NUM_UPDATES"]
        warmup_steps = int(config["LR_WARMUP"] * update_steps)
        steps_per_epoch = config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=base_lr,
            transition_steps=warmup_steps * steps_per_epoch,
        )
        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_lr,
            decay_steps=max(update_steps - warmup_steps, 1) * steps_per_epoch,
        )
        return optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_steps * steps_per_epoch],
        )

    def _dummy_obs():
        h, w, c = env.grid_shape
        m = env.msg_shape[0]
        return {
            "grid": jnp.zeros((1, config["NUM_ENVS"], h, w, c), dtype=jnp.float32),
            "last_allocation": jnp.zeros((1, config["NUM_ENVS"], m), dtype=jnp.float32),
            # is_t0 is a scalar-per-env obs; leading (T=1, N) mirrors the
            # trajectory layout the network consumes.
            "is_t0": jnp.zeros((1, config["NUM_ENVS"]), dtype=jnp.float32),
        }

    # Precompute Python labels for the capability pool. Used inside the jitted
    # trainer to build per-capability metric KEY names.
    _cap_pairs_py: List[tuple] = [
        (int(a), int(b)) for a, b in
        np.asarray(env.partner_capability_pairs).tolist()
    ]
    _n_partner_cap: int = len(_cap_pairs_py)

    # --- Build the (capability, layout) sample schedule -----------
    # Each entry = one 20-round partner episode with (a) a fixed
    # capability pair uniformly sampled from the training pool and
    # (b) an independent uniform layout for every round. See
    # episode_scheduler.build_schedule.
    n_eps_total = int(config.get(
        "N_EPS_TOTAL",
        max(int(config["NUM_ENVS"]) * 512, 8192),
    ))
    schedule = build_schedule(
        partner_capability_pairs=_cap_pairs_py,
        n_layouts_train=int(env.n_layouts),
        rounds_per_episode=int(env.rounds_per_episode),
        n_eps_total=n_eps_total,
        seed=int(config.get("SCHEDULE_SEED", config.get("SEED", 0))),
    )
    summarize_schedule(schedule, verbose=True)
    _SCHEDULE_CAP = jnp.asarray(schedule.schedule_capability, dtype=jnp.int32)  # (E_total, 2)
    _SCHEDULE_LAYOUTS = jnp.asarray(schedule.schedule_layouts,
                                     dtype=jnp.int32)                       # (E_total, R)
    _N_EPS_TOTAL = int(schedule.n_eps_total)

    def train(rng):
        # INIT NETWORK
        action_dim = env.n_ego_actions
        network, init_hstate_fn, _mtype = build_network(config, action_dim)

        rng, _rng = jax.random.split(rng)
        init_obs = _dummy_obs()
        init_dones = jnp.zeros((1, config["NUM_ENVS"]), dtype=bool)
        init_hstate = init_hstate_fn(config["NUM_ENVS"])
        network_params = network.init(_rng, init_hstate, (init_obs, init_dones))

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(create_learning_rate_fn(), eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply, params=network_params, tx=tx
        )

        # INIT ENVS — scheduled reset: each vmap slot starts on its own
        # sweep-boundary. Cursor increments by 1 per auto-reset per slot
        # and wraps at _N_EPS_TOTAL.
        cursor0 = jnp.asarray(
            initial_episode_cursor(config["NUM_ENVS"], schedule),
            dtype=jnp.int32,
        )                                                            # (N,)
        cap0 = _SCHEDULE_CAP[cursor0]                                # (N, 2)
        layouts0 = _SCHEDULE_LAYOUTS[cursor0]                        # (N, R)
        rng, k_reset = jax.random.split(rng)
        reset_keys = jax.random.split(k_reset, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset_from_schedule)(cap0, layouts0, reset_keys)

        # Per-slot episode-return / -length accumulators (LogWrapper-free).
        ep_return_acc0 = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.float32)
        ep_length_acc0 = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.int32)

        def _select_per_slot(a, b, mask_1d):
            """Per-slot `jnp.where(mask, b, a)` broadcasting `mask_1d`
            (shape (N,)) to whatever leading-N shape a/b have."""
            ndim_extra = a.ndim - 1
            m = mask_1d.reshape((-1,) + (1,) * ndim_extra)
            return jnp.where(m, b, a)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # --------- collect trajectories ---------
            def _env_step(runner_state, unused):
                (train_state, env_state, last_obs, last_done,
                 update_step, hstate, rng,
                 episode_cursor, ep_return_acc, ep_length_acc) = runner_state

                # Pre-step round-local time. Straight off state.time now that
                # we're not wrapping the env with LogWrapper.
                pre_step_time = env_state.time

                obs_agent0 = last_obs["agent_0"]
                obs_in = jax.tree_util.tree_map(
                    lambda x: x[jnp.newaxis, :], obs_agent0
                )
                dones_in = last_done[jnp.newaxis, :]

                hstate, pi, value = network.apply(
                    train_state.params, hstate, (obs_in, dones_in)
                )
                rng, _rng = jax.random.split(rng)
                action = pi.sample(seed=_rng)          # (1, N)
                log_prob = pi.log_prob(action)         # (1, N)

                env_act = {"agent_0": action.squeeze(0)}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                # env.step_env only — no autoreset. We handle reset manually.
                obsv, env_state_stepped, reward, done, info = jax.vmap(
                    env.step_env, in_axes=(0, 0, {"agent_0": 0})
                )(rng_step, env_state, env_act)

                reward_ego = reward["agent_0"]
                done_all = done["__all__"]

                # Force per-step info fields we care about into log-friendly dtype.
                info = dict(info)
                if "success" in info:
                    info["success"] = info["success"].astype(jnp.float32)

                # Episode-return / -length trackers.
                new_ret_acc = ep_return_acc + reward_ego
                new_len_acc = ep_length_acc + jnp.int32(1)
                # On terminal steps, log the finished-episode values (for
                # returned_episode_* info fields) BEFORE resetting the acc.
                returned_return = jnp.where(done_all, new_ret_acc,
                                             jnp.float32(0.0))
                returned_length = jnp.where(done_all, new_len_acc.astype(jnp.float32),
                                             jnp.float32(0.0))

                # Scheduled auto-reset: advance cursor, look up next (z, layouts).
                new_cursor = jnp.where(
                    done_all,
                    (episode_cursor + 1) % jnp.int32(_N_EPS_TOTAL),
                    episode_cursor,
                )
                next_cap = _SCHEDULE_CAP[new_cursor]             # (N, 2)
                next_layouts = _SCHEDULE_LAYOUTS[new_cursor]    # (N, R)
                rng, k_reset = jax.random.split(rng)
                reset_keys = jax.random.split(k_reset, config["NUM_ENVS"])
                reset_obs, reset_state = jax.vmap(env.reset_from_schedule)(
                    next_cap, next_layouts, reset_keys
                )
                # Per-slot select: for slots where done_all=True, use the
                # scheduled reset; else keep the stepped state.
                env_state_next = jax.tree_util.tree_map(
                    lambda a, b: _select_per_slot(a, b, done_all),
                    env_state_stepped, reset_state,
                )
                obsv_next = jax.tree_util.tree_map(
                    lambda a, b: _select_per_slot(a, b, done_all),
                    obsv, reset_obs,
                )
                # Reset accs on terminal slots.
                ep_return_acc_next = jnp.where(done_all,
                                                jnp.float32(0.0), new_ret_acc)
                ep_length_acc_next = jnp.where(done_all,
                                                jnp.int32(0), new_len_acc)

                # Add LogWrapper-compatible-ish info fields for the metric callback.
                info["returned_episode"] = done_all
                info["returned_episode_return"] = returned_return
                info["returned_episode_length"] = returned_length

                transition = Transition(
                    done=done_all,
                    reset=last_done,
                    action=action.squeeze(0),
                    value=value.squeeze(0),
                    reward=reward_ego,
                    log_prob=log_prob.squeeze(0),
                    obs=obs_agent0,
                    info=info,
                    pre_step_time=pre_step_time,
                )
                runner_state = (
                    train_state, env_state_next, obsv_next, done_all,
                    update_step, hstate, rng,
                    new_cursor, ep_return_acc_next, ep_length_acc_next,
                )
                return runner_state, transition

            # Snapshot hstate BEFORE the rollout — we replay from here in the loss.
            # runner_state layout is now:
            #   0: train_state
            #   1: env_state
            #   2: last_obs
            #   3: last_done
            #   4: update_step
            #   5: hstate
            #   6: rng
            #   7: episode_cursor
            #   8: ep_return_acc
            #   9: ep_length_acc
            initial_hstate = runner_state[5]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # --------- bootstrap value + GAE ---------
            (train_state, env_state, last_obs, last_done,
             update_step, hstate, rng,
             episode_cursor, ep_return_acc, ep_length_acc) = runner_state

            obs_agent0 = last_obs["agent_0"]
            obs_in = jax.tree_util.tree_map(
                lambda x: x[jnp.newaxis, :], obs_agent0
            )
            dones_in = last_done[jnp.newaxis, :]
            _, _, last_val = network.apply(
                train_state.params, hstate, (obs_in, dones_in)
            )
            last_val = last_val.squeeze(0)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done, transition.value, transition.reward
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # --------- PPO update ---------
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        _, pi, value = network.apply(
                            params,
                            init_hstate.squeeze(0),
                            (traj_batch.obs, traj_batch.reset),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean()

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        # Per-step entropy of the (already masked) categorical.
                        step_entropy = pi.entropy()                        # (T, N)
                        entropy = step_entropy.mean()

                        # Split entropy by t=0 vs t>=1 for logging. Uses the
                        # SAME is_t0 the mask was derived from, so t=0 entropy
                        # is zero (forced default), and t>=1 is capped at
                        # ln(10) (5 moves x 2 allocations).
                        is_t0_mask = (traj_batch.obs["is_t0"] > 0.5)       # (T, N)
                        n_t0 = is_t0_mask.sum()
                        n_tge1 = (~is_t0_mask).sum()
                        entropy_t0 = jnp.where(
                            n_t0 > 0,
                            (step_entropy * is_t0_mask).sum() / jnp.maximum(n_t0, 1),
                            0.0,
                        )
                        entropy_tge1 = jnp.where(
                            n_tge1 > 0,
                            (step_entropy * (~is_t0_mask)).sum() / jnp.maximum(n_tge1, 1),
                            0.0,
                        )

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (
                            value_loss, loss_actor, entropy,
                            entropy_t0, entropy_tge1,
                        )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch,
                        advantages, targets,
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (train_state, init_hstate, traj_batch,
                 advantages, targets, rng) = update_state
                rng, _rng = jax.random.split(rng)

                # Add leading time-dim = 1 to init_hstate so it minibatches
                # the same way as the T-leading traj tensors.
                init_hstate_batched = init_hstate[jnp.newaxis, :]
                batch = (
                    init_hstate_batched,
                    traj_batch,
                    advantages,
                    targets,
                )

                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1, 0,
                    ),
                    shuffled_batch,
                )

                train_state, loss_info = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state, init_hstate, traj_batch,
                    advantages, targets, rng,
                )
                return update_state, loss_info

            update_state = (
                train_state, initial_hstate, traj_batch,
                advantages, targets, rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            # --------- metrics ---------
            total_loss = loss_info[0]                      # (UPDATE_EPOCHS, NUM_MINIBATCHES)
            (value_loss, actor_loss, entropy,
             entropy_t0, entropy_tge1) = loss_info[1]
            new_update_step = update_step + 1

            # ---- Round-level success rate ----
            # A partner episode contains rounds_per_episode rounds and
            # `traj_batch.done` (== done["__all__"]) only fires at the end of
            # the final round. Round-level success is the right per-round
            # metric: mask successes by info["round_done"] and average.
            round_done = traj_batch.info.get(
                "round_done", jnp.zeros_like(traj_batch.done)
            ).astype(jnp.float32)                                            # (T, N)
            successes = traj_batch.info.get(
                "success", jnp.zeros_like(traj_batch.done)
            ).astype(jnp.float32)                                            # (T, N)
            n_rounds = round_done.sum()
            round_success_rate = jnp.where(
                n_rounds > 0,
                (successes * round_done).sum() / jnp.maximum(n_rounds, 1),
                0.0,
            )

            # Partner-episode-level: whole-episode success counts only end-of-
            # -episode steps — the fraction of partner episodes whose final
            # round succeeded (noisier than the per-round rate above).
            episode_done = traj_batch.done.astype(jnp.float32)
            n_completed = episode_done.sum()
            partner_ep_success_rate = jnp.where(
                n_completed > 0,
                (successes * episode_done).sum() / jnp.maximum(n_completed, 1),
                0.0,
            )

            # Random initialization distribution, not a learned t=0 allocation:
            # (a = 3*move + alloc), mask to steps where pre_step_time == 0.
            # alloc ids: 0=NONE (never legal), 1=ALLOC_RED, 2=ALLOC_BLUE.
            ego_alloc_all = traj_batch.action % 3                            # (T, N) int
            is_t0 = (traj_batch.pre_step_time == 0).astype(jnp.float32)      # (T, N)
            n_t0 = is_t0.sum()
            frac_alloc_red = jnp.where(
                n_t0 > 0, ((ego_alloc_all == 1) * is_t0).sum() / jnp.maximum(n_t0, 1), 0.0)
            frac_alloc_blue = jnp.where(
                n_t0 > 0, ((ego_alloc_all == 2) * is_t0).sum() / jnp.maximum(n_t0, 1), 0.0)

            # ---- Per-capability aggregates ----
            # traj_batch.info["capability"] has shape (T, N, 2) — capability is
            # broadcast over time within a partner episode. Group round_done
            # rows by capability pair.
            cap_traj = traj_batch.info.get(
                "capability", jnp.zeros(traj_batch.reward.shape + (2,), dtype=jnp.int32)
            )                                                    # (T, N, 2)
            per_cap_success = []
            per_cap_rounds = []
            per_cap_t0_alloc_red = []
            for ci, cap_py in enumerate(_cap_pairs_py):
                cr = jnp.int32(cap_py[0]); cb = jnp.int32(cap_py[1])
                cap_mask = (
                    (cap_traj[..., 0] == cr) & (cap_traj[..., 1] == cb)
                ).astype(jnp.float32)                            # (T, N)
                rd_c = round_done * cap_mask
                n_rd_c = rd_c.sum()
                succ_c = jnp.where(
                    n_rd_c > 0,
                    (successes * rd_c).sum() / jnp.maximum(n_rd_c, 1),
                    0.0,
                )
                per_cap_success.append(succ_c)
                per_cap_rounds.append(n_rd_c)
                t0_c = is_t0 * cap_mask
                n_t0_c = t0_c.sum()
                fred = jnp.where(
                    n_t0_c > 0,
                    ((ego_alloc_all == 1) * t0_c).sum() / jnp.maximum(n_t0_c, 1),
                    0.0,
                )
                per_cap_t0_alloc_red.append(fred)

            metric = {
                "round_success_rate": round_success_rate,
                "rounds_completed": n_rounds,
                "partner_ep_success_rate": partner_ep_success_rate,
                "partner_eps_completed": n_completed,
                "success_rate": round_success_rate,
                "reward_mean": traj_batch.reward.mean(),
                "value_loss": value_loss.mean(),
                "actor_loss": actor_loss.mean(),
                "entropy": entropy.mean(),
                "entropy_t0": entropy_t0.mean(),      # forced default: zero
                "entropy_tge1": entropy_tge1.mean(),  # 10 choices, max=ln 10
                "total_loss": total_loss.mean(),
                "initial_alloc_red": frac_alloc_red,
                "initial_alloc_blue": frac_alloc_blue,
                "online_alloc_red": (((ego_alloc_all == 1) * (1-is_t0)).sum()
                                     / jnp.maximum((1-is_t0).sum(), 1)),
                "assignment_switch_rate": (traj_batch.info["assignment_changed"].sum()
                                           / jnp.maximum((1-is_t0).sum(), 1)),
                "t0_steps_seen": n_t0,
                "update_step": new_update_step,
                "env_step": new_update_step
                            * config["NUM_STEPS"] * config["NUM_ENVS"],
            }
            # Per-capability metrics (one key per pair). Small enough set that
            # keying every pair is fine for W&B.
            for ci, cap_py in enumerate(_cap_pairs_py):
                cap_label = f"{cap_py[0]}-{cap_py[1]}"
                metric[f"cap={cap_label}/round_success"] = per_cap_success[ci]
                metric[f"cap={cap_label}/rounds_seen"] = per_cap_rounds[ci]
                metric[f"cap={cap_label}/initial_alloc_red"] = per_cap_t0_alloc_red[ci]
            # Episode-return / -length written by the trainer's manual
            # auto-reset. returned_episode is True only on the step an
            # episode ends; the return/length payload is 0.0 elsewhere.
            if "returned_episode_return" in traj_batch.info:
                ret_mask = traj_batch.info["returned_episode"].astype(jnp.float32)
                ret_val = traj_batch.info["returned_episode_return"]
                ret_len = traj_batch.info["returned_episode_length"]
                n_ret = ret_mask.sum()
                metric["ep_return_mean"] = jnp.where(
                    n_ret > 0, (ret_val * ret_mask).sum() / jnp.maximum(n_ret, 1), 0.0
                )
                metric["ep_length_mean"] = jnp.where(
                    n_ret > 0, (ret_len * ret_mask).sum() / jnp.maximum(n_ret, 1), 0.0
                )

            def _log_cb(m):
                # Cast anything jax-flavoured to python scalars for wandb.
                flat = {k: (float(v) if hasattr(v, "shape") else v)
                        for k, v in m.items()}
                wandb.log(flat)
                if config.get("STDOUT_LOG", False):
                    ep_ret = flat.get("ep_return_mean", float("nan"))
                    ep_len = flat.get("ep_length_mean", float("nan"))
                    print(
                        f"[u {int(flat['update_step']):>4d}] "
                        f"env={int(flat['env_step']):>9d} "
                        f"rsucc={flat['round_success_rate']:.3f} "
                        f"n_rounds={int(flat['rounds_completed']):>5d} "
                        f"ret={ep_ret:+.3f} len={ep_len:4.1f} "
                        f"ent(t0,t>=1)=({flat['entropy_t0']:.3f},"
                        f"{flat['entropy_tge1']:.3f}) "
                        f"aL={flat['actor_loss']:+.4f} vL={flat['value_loss']:.4f} "
                        f"initial=[red={flat['initial_alloc_red']:.2f} "
                        f"blue={flat['initial_alloc_blue']:.2f}] "
                        f"switch={flat['assignment_switch_rate']:.3f}",
                        flush=True,
                    )
            jax.debug.callback(_log_cb, metric)

            runner_state = (
                train_state, env_state, last_obs, last_done,
                new_update_step, hstate, rng,
                episode_cursor, ep_return_acc, ep_length_acc,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            0,
            init_hstate,
            _rng,
            cursor0,
            ep_return_acc0,
            ep_length_acc0,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


# ---------------------------------------------------------------------------
# Held-out evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(params, config, layouts_dir, key,
                    n_episodes_per_capability: int = 20,
                    capability_pairs_override: Optional[Sequence[Sequence[int]]] = None):
    """Run ``n_episodes_per_capability`` full partner-episodes per capability
    pair, on the given layout pool (this experiment uses the SAME 1000
    layouts as training). Reports round-level success plus per-capability
    and per-round-index breakdowns.

    ``capability_pairs_override`` — evaluate against this explicit pool
    (e.g. training or novel test pairs).
    """
    require_current_protocol(config)
    env_kwargs = dict(config["ENV_KWARGS"])
    env_kwargs["layouts_dir"] = layouts_dir
    env_kwargs["augment_symmetries"] = False
    env_kwargs.pop("layout_path", None)
    env_kwargs.pop("layout_paths", None)
    if capability_pairs_override is not None:
        env_kwargs["partner_capability_pairs"] = [
            [int(a), int(b)] for a, b in capability_pairs_override
        ]
    env_eval = CoordinationGrid(**env_kwargs)

    if env_eval.communication_condition != COMM_ACTION_ONLY:
        raise ValueError(
            f"eval env communication_condition must be 'action_only', "
            f"got {env_eval.communication_condition!r}"
        )
    network, init_hstate_fn, _ = build_network(config, env_eval.n_ego_actions)
    C = int(env_eval.n_partner_capability)
    N = int(n_episodes_per_capability)
    B = C * N
    R = int(env_eval.rounds_per_episode)
    T = R * env_eval.max_steps

    # (B,) capability assignment: N episodes per capability pair.
    cap_index_per_ep = jnp.repeat(jnp.arange(C, dtype=jnp.int32), N)
    cap_per_ep = env_eval.partner_capability_pairs[cap_index_per_ep]  # (B, 2)

    key_reset, key_step = jax.random.split(key)
    reset_keys = jax.random.split(key_reset, B)
    obs, states = jax.vmap(env_eval.reset)(reset_keys)
    # Override the randomly-sampled capability with our per-slot pair.
    states = states.replace(capability=cap_per_ep)
    obs = jax.vmap(env_eval.get_obs)(states)

    hstate0 = init_hstate_fn(B)
    done_prev0 = jnp.zeros((B,), dtype=bool)

    @jax.jit
    def rollout(params, obs, states, hstate, done_prev, key):
        def body(carry, _):
            obs, states, hstate, done_prev, key = carry
            obs_agent0 = obs["agent_0"]
            obs_in = jax.tree_util.tree_map(
                lambda x: x[jnp.newaxis, :], obs_agent0
            )
            done_in = done_prev[jnp.newaxis, :]
            hstate, pi, _ = network.apply(params, hstate, (obs_in, done_in))
            key, ka, ks = jax.random.split(key, 3)
            action = pi.sample(seed=ka).squeeze(0)                 # (B,)
            step_keys = jax.random.split(ks, B)
            obs, states, reward, done, info = jax.vmap(
                env_eval.step_env, in_axes=(0, 0, {"agent_0": 0})
            )(step_keys, states, {"agent_0": action})
            done_all = done["__all__"]
            return (obs, states, hstate, done_all, key), (
                reward["agent_0"],
                done_all,
                info["success"].astype(jnp.float32),
                info["round_done"].astype(jnp.float32),
                info["round_idx"].astype(jnp.int32),
            )

        init_carry = (obs, states, hstate, done_prev, key)
        _, out = jax.lax.scan(body, init_carry, None, length=T)
        return out

    (rewards, dones, successes,
     round_dones, round_idxs) = rollout(
        params, obs, states, hstate0, done_prev0, key_step
    )
    dones_np = np.asarray(dones)
    first_done_idx = np.argmax(dones_np.astype(np.int32), axis=0)
    never_done = ~dones_np.any(axis=0)
    T_ax = np.arange(dones_np.shape[0])[:, None]
    alive_mask_np = (
        (T_ax <= first_done_idx[None, :]) | never_done[None, :]
    ).astype(np.float32)
    alive_mask = jnp.asarray(alive_mask_np)

    round_dones_alive = round_dones * alive_mask
    n_rounds_total = round_dones_alive.sum()
    round_success_overall = float(
        (successes * round_dones_alive).sum() / jnp.maximum(n_rounds_total, 1)
    )
    ep_return = (rewards * alive_mask).sum(axis=0)

    per_cap_success = np.zeros(C, dtype=np.float32)
    per_cap_return  = np.zeros(C, dtype=np.float32)
    per_cap_rounds  = np.zeros(C, dtype=np.int64)
    cap_idx_np = np.asarray(cap_index_per_ep)
    rd_alive_np = np.asarray(round_dones_alive)
    s_np        = np.asarray(successes)
    for ci in range(C):
        cols = (cap_idx_np == ci)
        rd_c = rd_alive_np[:, cols]
        s_c  = s_np[:, cols]
        nrd  = int(rd_c.sum())
        per_cap_success[ci] = float((s_c * rd_c).sum() / max(nrd, 1))
        per_cap_return[ci]  = float(np.asarray(ep_return)[cols].mean())
        per_cap_rounds[ci]  = nrd

    per_round_success = np.zeros(R, dtype=np.float32)
    per_round_n       = np.zeros(R, dtype=np.int64)
    r_idx_np = np.asarray(round_idxs)
    for r in range(R):
        mask = (r_idx_np == r) & (rd_alive_np > 0.5)
        n = int(mask.sum())
        if n > 0:
            per_round_success[r] = float(s_np[mask].sum() / n)
        per_round_n[r] = n

    cap_pairs_out = [
        [int(a), int(b)] for a, b in np.asarray(env_eval.partner_capability_pairs)
    ]
    return {
        "n_episodes":                      int(B),
        "n_episodes_per_capability":       int(N),
        "n_partner_capability":            int(C),
        "rounds_per_episode":              int(R),
        "n_layouts_pool":                  int(env_eval.n_layouts),
        "partner_capability_pairs":        cap_pairs_out,
        "round_success_overall":           round_success_overall,
        "n_rounds_total":                  int(n_rounds_total),
        "per_capability_success":          per_cap_success.tolist(),
        "per_capability_return":           per_cap_return.tolist(),
        "per_capability_rounds":           per_cap_rounds.tolist(),
        "per_round_idx_success":           per_round_success.tolist(),
        "per_round_idx_n":                 per_round_n.tolist(),
        "mean_ep_return":                  float(np.asarray(ep_return).mean()),
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path="config",
    config_name="ippo_coordination_grid",
)
def main(config):
    config = OmegaConf.to_container(config)
    num_seeds = config["NUM_SEEDS"]
    start_time = datetime.now()

    model_type = str(config.get("MODEL_TYPE", "rnn")).lower()
    regime = str(config.get("PARTNER_REGIME", "diverse")).lower()
    influence = bool(config.get("INFLUENCE", True))
    wandb.init(
        entity=config.get("ENTITY", ""),
        project=config.get("PROJECT", "coordination_grid"),
        tags=["IPPO", model_type.upper(), "CoordinationGrid",
              f"regime={regime}", f"influence={influence}"],
        config=config,
        mode=config.get("WANDB_MODE", "disabled"),
        name=(
            f"ippo_{model_type}_{regime}"
            f"_{'inf' if influence else 'noinf'}"
            f"_R{config['ENV_KWARGS'].get('rounds_per_episode', 1)}"
            f"_seed{config['SEED']}"
        ),
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, num_seeds)
    print(f"[main] compiling and running {num_seeds} seed(s)...", flush=True)
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)
    # Block until all async work has resolved so wall-clock is honest.
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        out["metrics"],
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    wandb.log({"wallclock_seconds": elapsed})
    print(f"[main] training done in {elapsed:.1f}s", flush=True)

    # Print a summary if requested (smoke tests / no W&B).
    if config.get("STDOUT_SUMMARY", False):
        m = out["metrics"]
        for k in ("round_success_rate", "partner_ep_success_rate",
                  "ep_return_mean", "ep_length_mean",
                  "reward_mean", "actor_loss", "value_loss",
                  "entropy", "entropy_t0", "entropy_tge1",
                  "initial_alloc_red", "initial_alloc_blue", "online_alloc_red",
                  "assignment_switch_rate"):
            if k in m:
                v = np.asarray(m[k])
                seq = v.mean(axis=0)
                nice = ", ".join(f"{x:+.4f}" for x in seq.tolist())
                print(f"    {k:>22s}: [{nice}]", flush=True)

    # -------------- Save params (per seed, seed 0 as canonical) --------------
    # out["runner_state"][0] is the TrainState (vmapped over seeds).
    save_path = config.get("SAVE_PARAMS_PATH", "")
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        params_all_seeds = out["runner_state"][0].params
        # For the canonical dump, take seed 0.
        params_seed0 = jax.tree_util.tree_map(lambda x: x[0], params_all_seeds)
        save_params(params_seed0, save_path)
        print(f"[main] saved params (seed 0) -> {save_path}", flush=True)
        # If multiple seeds, also drop per-seed dumps next to it.
        if num_seeds > 1:
            base, ext = os.path.splitext(save_path)
            for s in range(num_seeds):
                ps = jax.tree_util.tree_map(lambda x, i=s: x[i], params_all_seeds)
                path_s = f"{base}_seed{s}{ext}"
                save_params(ps, path_s)
            print(f"[main] saved {num_seeds} per-seed dumps beside {save_path}",
                  flush=True)

        # Save the fully resolved training config next to the checkpoint so
        # the standalone evaluator can reconstruct the environment exactly
        # (MODEL_TYPE, PARTNER_REGIME, INFLUENCE, SINGLE_PARTNER, ENV_KWARGS,
        # PPO hyperparameters, network sizes, schedule params, etc.).
        # The evaluator loads this JSON with --config.
        import json as _json_cfg
        config_out_path = os.path.splitext(save_path)[0] + "_config.json"
        cfg_serializable = _json_cfg.loads(
            _json_cfg.dumps(config, default=str)
        )
        with open(config_out_path, "w") as f:
            _json_cfg.dump(cfg_serializable, f, indent=2)
        print(f"[main] saved resolved config -> {config_out_path}", flush=True)

    # -------------- Eval on the SAME 1000-layout corpus as training --------------
    # Two capability slices are reported:
    #   train : the exact training pool (familiar profiles)
    #   test  : the novel-partner test population from
    #           capability_populations.TEST_CAPABILITY_PAIRS — includes
    #           scalar delay values (0, 5, 6) never seen at training.
    # The primary scientific result is the test slice. This runs a
    # lightweight in-process eval; the full evaluation lives in
    # eval/evaluate_partner_modelling.py.
    layouts_dir = config["ENV_KWARGS"].get("layouts_dir", None)
    n_eps = int(config.get("EVAL_EPISODES_PER_CAPABILITY", 20))
    train_cap_pool = [list(p) for p in TRAIN_CAPABILITY_PAIRS]
    test_cap_pool = [list(p) for p in TEST_CAPABILITY_PAIRS]
    if layouts_dir:
        params_seed0 = jax.tree_util.tree_map(
            lambda x: x[0], out["runner_state"][0].params
        )
        rng_eval = jax.random.PRNGKey(int(config.get("EVAL_SEED", 12345)))
        eval_summary: Dict[str, dict] = {}
        for cap_slice_name, cap_pool in (
            ("train", train_cap_pool),
            ("test",  test_cap_pool),
        ):
            rng_eval, sub = jax.random.split(rng_eval)
            print(
                f"[eval] {cap_slice_name}: layouts_dir={layouts_dir}, "
                f"{n_eps} episodes/cap × {len(cap_pool)} capability pairs...",
                flush=True,
            )
            r = evaluate_policy(
                params_seed0, config, layouts_dir, sub,
                n_episodes_per_capability=n_eps,
                capability_pairs_override=cap_pool,
            )
            eval_summary[cap_slice_name] = r
            print(
                f"[eval] {cap_slice_name}: "
                f"round_success_overall={r['round_success_overall']:.3f}"
                f"  (n_rounds={r['n_rounds_total']})",
                flush=True,
            )
            for cap, sv, nrv in zip(
                r["partner_capability_pairs"],
                r["per_capability_success"],
                r["per_capability_rounds"],
            ):
                print(
                    f"    cap=({cap[0]},{cap[1]})  succ={sv:.3f}  "
                    f"n_rounds={nrv}", flush=True,
                )
            wandb.log({
                f"eval_{cap_slice_name}/round_success_overall":
                    r["round_success_overall"],
                **{f"eval_{cap_slice_name}/per_cap_success/cap={cap[0]}-{cap[1]}": sv
                   for cap, sv in zip(
                       r["partner_capability_pairs"],
                       r["per_capability_success"],
                   )},
            })
        if save_path:
            eval_json_path = os.path.splitext(save_path)[0] + "_eval.json"
            import json as _json
            with open(eval_json_path, "w") as f:
                _json.dump(eval_summary, f, indent=2)
            print(f"[eval] wrote per-capability / per-round numbers -> "
                  f"{eval_json_path}", flush=True)

    return out


if __name__ == "__main__":
    main()
