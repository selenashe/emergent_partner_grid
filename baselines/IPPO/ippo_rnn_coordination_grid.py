"""Recurrent IPPO for CoordinationGrid — Stage A.

Only agent_0 is a learned actor; the scripted partner runs inside the env.
Obs is a dict {"grid": (H, W, 5), "last_message": (3,)}: the grid goes through
a CNN, the last_message through a small dense embedding, and both are
concatenated before the GRU. Actor head is a flat 15-way Categorical over the
factored (move, message) ego action.

Forked from ``ippo_rnn_overcooked_v2.py`` with all Overcooked-specific
scaffolding removed: reward shaping, wait-buffer, multi-network experiments,
checkpoint loading, second-agent stacking, batchify/unbatchify. W&B stays
wired but defaults to disabled.

Stage A scope:
    * one fixed layout
    * fixed partner_z (typically 1.0)
    * one round per episode (`done["__all__"]` from the env is terminal)

Deferred to later stages (per plan): factored action head, layout variation,
round_done ≠ episode_done + per-partner z sampling.
"""

import functools
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, NamedTuple, Sequence

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
    ACTION_MASK_T0, ACTION_MASK_TGEQ1,
)


# Static mask tensors (bool) for the network. jnp arrays so they broadcast
# cleanly with logit tensors of arbitrary leading shape.
_ACTION_MASK_T0_J = jnp.asarray(ACTION_MASK_T0, dtype=jnp.bool_)      # (15,)
_ACTION_MASK_TGEQ1_J = jnp.asarray(ACTION_MASK_TGEQ1, dtype=jnp.bool_)  # (15,)


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
    last_message: (N, 3) -> Dense -> (N, msg_dim)
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
        msg = obs["last_message"].astype(jnp.float32)

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

        # Legality mask: state-dependent, drives sample / log_prob / entropy.
        # is_t0 has leading (T, N) shape (scalar per env-step); broadcast against
        # the two (15,) masks and set illegal logits to -inf so distrax's
        # Categorical excludes them from the softmax.
        is_t0 = obs["is_t0"]                                          # (T, N)
        is_t0_bool = (is_t0 > 0.5)[..., None]                         # (T, N, 1)
        legal = jnp.where(
            is_t0_bool,
            _ACTION_MASK_T0_J[None, None, :],
            _ACTION_MASK_TGEQ1_J[None, None, :],
        )                                                             # (T, N, 15)
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


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: Any  # dict pytree {"grid": ..., "last_message": ...}
    info: Dict
    # env-state fields captured for logging (BEFORE step_env ran on this step)
    pre_step_time: jnp.ndarray  # (N,) int32


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # ONE learned actor per env (partner is scripted inside the env).
    config["NUM_ACTORS"] = config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env, replace_info=False)

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
            "last_message": jnp.zeros((1, config["NUM_ENVS"], m), dtype=jnp.float32),
            # is_t0 is a scalar-per-env obs; leading (T=1, N) mirrors the
            # trajectory layout the network consumes.
            "is_t0": jnp.zeros((1, config["NUM_ENVS"]), dtype=jnp.float32),
        }

    # Precompute Python-float labels for the z pool. Used inside the jitted
    # trainer to build per-z metric KEY names — those keys can't come from
    # `float(env.partner_z_values[zi])` under trace (traced concretization).
    _z_values_py: List[float] = [
        float(v) for v in np.asarray(env.partner_z_values).tolist()
    ]
    _n_partner_z: int = len(_z_values_py)

    def train(rng):
        # INIT NETWORK
        action_dim = env.n_ego_actions
        network = ActorCriticCommRNN(action_dim=action_dim, config=config)

        rng, _rng = jax.random.split(rng)
        init_obs = _dummy_obs()
        init_dones = jnp.zeros((1, config["NUM_ENVS"]), dtype=bool)
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
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

        # INIT ENVS
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # --------- collect trajectories ---------
            def _env_step(runner_state, unused):
                (train_state, env_state, last_obs, last_done,
                 update_step, hstate, rng) = runner_state

                # Grab the ENV's pre-step time. LogWrapper wraps the env so the
                # underlying CoordinationGrid state (with .time) lives at
                # env_state.env_state. This is 0 on the free-comm step of every
                # episode — which is exactly the mask we need to condition the
                # t=0 message distribution on.
                pre_step_time = env_state.env_state.time

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
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                reward_ego = reward["agent_0"]
                done_all = done["__all__"]

                # Force per-step info fields we care about into log-friendly dtype.
                info = dict(info)
                if "success" in info:
                    info["success"] = info["success"].astype(jnp.float32)

                transition = Transition(
                    done=done_all,
                    action=action.squeeze(0),
                    value=value.squeeze(0),
                    reward=reward_ego,
                    log_prob=log_prob.squeeze(0),
                    obs=obs_agent0,
                    info=info,
                    pre_step_time=pre_step_time,
                )
                runner_state = (
                    train_state, env_state, obsv, done_all,
                    update_step, hstate, rng,
                )
                return runner_state, transition

            # Snapshot hstate BEFORE the rollout — we replay from here in the loss.
            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # --------- bootstrap value + GAE ---------
            (train_state, env_state, last_obs, last_done,
             update_step, hstate, rng) = runner_state

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
                            (traj_batch.obs, traj_batch.done),
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
                        # is capped at ln(3) and t>=1 at ln(5).
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
            # Under Stage C+D, a *partner episode* contains many rounds and
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
            # -episode steps. In Stage A/B (rounds_per_episode=1) this equals
            # round_success_rate; in Stage C+D it's the fraction of *partner
            # episodes* whose final round succeeded (a much noisier number).
            episode_done = traj_batch.done.astype(jnp.float32)
            n_completed = episode_done.sum()
            partner_ep_success_rate = jnp.where(
                n_completed > 0,
                (successes * episode_done).sum() / jnp.maximum(n_completed, 1),
                0.0,
            )

            # t=0 message distribution: decode msg from stored flat action
            # (a = 3*move + msg), mask to steps where pre_step_time == 0.
            ego_msg_all = traj_batch.action % 3                              # (T, N) int
            is_t0 = (traj_batch.pre_step_time == 0).astype(jnp.float32)      # (T, N)
            n_t0 = is_t0.sum()
            frac_msg_none = jnp.where(
                n_t0 > 0, ((ego_msg_all == 0) * is_t0).sum() / jnp.maximum(n_t0, 1), 0.0)
            frac_msg_m0 = jnp.where(
                n_t0 > 0, ((ego_msg_all == 1) * is_t0).sum() / jnp.maximum(n_t0, 1), 0.0)
            frac_msg_m1 = jnp.where(
                n_t0 > 0, ((ego_msg_all == 2) * is_t0).sum() / jnp.maximum(n_t0, 1), 0.0)

            # ---- Per-z aggregates (rolls up round-success and t0 msg by z) ----
            # traj_batch.info["z"] has shape (T, N) — z is broadcast over time
            # within a partner episode. Group round_done rows by z value.
            # `_z_values_py` are concrete Python floats known at trace time so
            # we can index and format them into metric keys under jit.
            z_traj = traj_batch.info.get("z", jnp.zeros_like(traj_batch.reward))  # (T, N)
            per_z_success = []
            per_z_rounds = []
            per_z_t0_msg_m0 = []
            per_z_t0_msg_m1 = []
            for zi, z_py in enumerate(_z_values_py):
                # Steps whose partner-episode has this z.
                z_mask = jnp.isclose(z_traj, jnp.float32(z_py), atol=1e-6).astype(jnp.float32)
                rd_z = round_done * z_mask                                     # round-done AND this z
                n_rd_z = rd_z.sum()
                succ_z = jnp.where(
                    n_rd_z > 0,
                    (successes * rd_z).sum() / jnp.maximum(n_rd_z, 1),
                    0.0,
                )
                per_z_success.append(succ_z)
                per_z_rounds.append(n_rd_z)
                # t0 msg dist within this z's partner episodes
                t0_z = is_t0 * z_mask
                n_t0_z = t0_z.sum()
                fm0 = jnp.where(
                    n_t0_z > 0,
                    ((ego_msg_all == 1) * t0_z).sum() / jnp.maximum(n_t0_z, 1),
                    0.0,
                )
                fm1 = jnp.where(
                    n_t0_z > 0,
                    ((ego_msg_all == 2) * t0_z).sum() / jnp.maximum(n_t0_z, 1),
                    0.0,
                )
                per_z_t0_msg_m0.append(fm0)
                per_z_t0_msg_m1.append(fm1)

            metric = {
                # Round-level (primary Stage C+D metric)
                "round_success_rate": round_success_rate,
                "rounds_completed": n_rounds,
                # Partner-episode-level (final round only)
                "partner_ep_success_rate": partner_ep_success_rate,
                "partner_eps_completed": n_completed,
                # Back-compat alias — some earlier logging / plotting reads this;
                # under Stage A/B (rounds_per_episode=1) it equals round success.
                "success_rate": round_success_rate,
                "reward_mean": traj_batch.reward.mean(),
                "value_loss": value_loss.mean(),
                "actor_loss": actor_loss.mean(),
                "entropy": entropy.mean(),           # combined masked entropy
                "entropy_t0": entropy_t0.mean(),      # over 3 msg options, max=ln 3
                "entropy_tge1": entropy_tge1.mean(),  # over 5 move options, max=ln 5
                "total_loss": total_loss.mean(),
                "t0_msg_none": frac_msg_none,
                "t0_msg_m0": frac_msg_m0,
                "t0_msg_m1": frac_msg_m1,
                "t0_steps_seen": n_t0,
                "update_step": new_update_step,
                "env_step": new_update_step
                            * config["NUM_STEPS"] * config["NUM_ENVS"],
            }
            # Per-z metrics (one key per z value, so W&B can chart them).
            for zi, z_py in enumerate(_z_values_py):
                z_label = f"{z_py:.2f}"
                metric[f"z={z_label}/round_success"] = per_z_success[zi]
                metric[f"z={z_label}/rounds_seen"] = per_z_rounds[zi]
                metric[f"z={z_label}/t0_msg_m0"] = per_z_t0_msg_m0[zi]
                metric[f"z={z_label}/t0_msg_m1"] = per_z_t0_msg_m1[zi]
            if "returned_episode_returns" in traj_batch.info:
                # LogWrapper writes returned_episode_* every step; the mask
                # returned_episode is True only on the step an episode ends.
                # Ego is at agent index 0.
                ret_mask = traj_batch.info["returned_episode"][..., 0].astype(jnp.float32)
                ret_val = traj_batch.info["returned_episode_returns"][..., 0]
                ret_len = traj_batch.info["returned_episode_lengths"][..., 0]
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
                        f"t0=[none={flat['t0_msg_none']:.2f} "
                        f"m0={flat['t0_msg_m0']:.2f} m1={flat['t0_msg_m1']:.2f}]",
                        flush=True,
                    )
            jax.debug.callback(_log_cb, metric)

            runner_state = (
                train_state, env_state, last_obs, last_done,
                new_update_step, hstate, rng,
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
                    n_episodes_per_z: int = 64):
    """Run ``n_episodes_per_z`` full partner-episodes per z value in the pool,
    on held-out layouts. Reports round-level success (primary Stage C+D
    metric), plus per-z and per-round-index breakdowns.

    Structure of the rollout:
      * Build an eval env with the training pool's ``partner_z_values`` and
        ``rounds_per_episode`` (default 20). Layout sampling per round is
        internal to the env; we do NOT ``reset_to_layout`` here because a
        partner episode spans MULTIPLE layouts (one per round).
      * Force each vmap slot to a specific z by manually setting state.z after
        reset — so we can group results by z without relying on random draws.
      * Rollout for ``rounds_per_episode * max_steps`` steps (exact horizon).
    """
    env_kwargs = dict(config["ENV_KWARGS"])
    # Eval always uses the raw val/test layouts.
    env_kwargs["layouts_dir"] = layouts_dir
    env_kwargs["augment_symmetries"] = False
    env_kwargs.pop("layout_path", None)
    env_kwargs.pop("layout_paths", None)
    env_eval = CoordinationGrid(**env_kwargs)

    network = ActorCriticCommRNN(
        action_dim=env_eval.n_ego_actions, config=config
    )
    Z = int(env_eval.n_partner_z)
    N = int(n_episodes_per_z)
    B = Z * N
    R = int(env_eval.rounds_per_episode)
    T = R * env_eval.max_steps  # rollout horizon that always contains 1 partner episode

    # (B,) z assignment: N episodes per z.
    z_index_per_ep = jnp.repeat(jnp.arange(Z, dtype=jnp.int32), N)
    z_per_ep = env_eval.partner_z_values[z_index_per_ep]           # (B,)

    key_reset, key_step = jax.random.split(key)
    reset_keys = jax.random.split(key_reset, B)
    obs, states = jax.vmap(env_eval.reset)(reset_keys)
    # Override the randomly-sampled z with our per-slot z, keep everything
    # else. layout_idx (sampled at reset) is fine to keep — layout will vary
    # across rounds anyway.
    states = states.replace(z=z_per_ep)
    obs = jax.vmap(env_eval.get_obs)(states)

    hstate0 = ScannedRNN.initialize_carry(B, config["GRU_HIDDEN_DIM"])
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
            # NOTE: use env.step_env (not env.step). We want a single partner
            # episode with round transitions handled internally, but NO auto
            # reset — otherwise our per-slot z override gets clobbered when
            # the final round ends. The horizon is set to exactly one episode.
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
    # shapes: (T, B) each. NOTE: this scan uses env.step_env (not env.step),
    # so once done["__all__"] fires there is NO autoreset. The env's stuck
    # state continues to report round_done=True on each subsequent call
    # because state.time keeps advancing past max_steps. Mask those out.
    dones_np = np.asarray(dones)                                    # (T, B) bool
    first_done_idx = np.argmax(dones_np.astype(np.int32), axis=0)   # (B,)
    # If done never fires within the horizon (shouldn't happen: T = R*max_steps
    # guarantees the episode ends), argmax returns 0 by default; guard by
    # taking all steps as alive in that case.
    never_done = ~dones_np.any(axis=0)                              # (B,)
    T_ax = np.arange(dones_np.shape[0])[:, None]                    # (T, 1)
    alive_mask_np = (
        (T_ax <= first_done_idx[None, :]) | never_done[None, :]
    ).astype(np.float32)                                            # (T, B)
    alive_mask = jnp.asarray(alive_mask_np)

    # ---- Round-level aggregation (masked to alive steps only) ----
    round_dones_alive = round_dones * alive_mask
    n_rounds_total = round_dones_alive.sum()
    round_success_overall = float(
        (successes * round_dones_alive).sum() / jnp.maximum(n_rounds_total, 1)
    )
    # Per-episode return: sum of masked rewards over the (alive) horizon.
    ep_return = (rewards * alive_mask).sum(axis=0)                  # (B,)

    # Per z (each z has N episodes; each ep contributes R rounds if not
    # short-circuited by autoreset — with step_env only, always R rounds).
    per_z_success = np.zeros(Z, dtype=np.float32)
    per_z_return  = np.zeros(Z, dtype=np.float32)
    per_z_rounds  = np.zeros(Z, dtype=np.int64)
    z_idx_np = np.asarray(z_index_per_ep)
    rd_alive_np = np.asarray(round_dones_alive)                      # (T, B)
    s_np        = np.asarray(successes)                              # (T, B)
    for zi in range(Z):
        cols = (z_idx_np == zi)
        rd_z = rd_alive_np[:, cols]                                  # (T, N)
        s_z  = s_np[:, cols]
        nrd  = int(rd_z.sum())
        per_z_success[zi] = float((s_z * rd_z).sum() / max(nrd, 1))
        per_z_return[zi]  = float(np.asarray(ep_return)[cols].mean())
        per_z_rounds[zi]  = nrd

    # Per round-index within episode (does success rise across rounds?).
    per_round_success = np.zeros(R, dtype=np.float32)
    per_round_n       = np.zeros(R, dtype=np.int64)
    r_idx_np = np.asarray(round_idxs)                                # (T, B)
    for r in range(R):
        mask = (r_idx_np == r) & (rd_alive_np > 0.5)
        n = int(mask.sum())
        if n > 0:
            per_round_success[r] = float(s_np[mask].sum() / n)
        per_round_n[r] = n

    return {
        "n_episodes":         int(B),
        "n_episodes_per_z":   int(N),
        "n_partner_z":        int(Z),
        "rounds_per_episode": int(R),
        "n_layouts_pool":     int(env_eval.n_layouts),
        "partner_z_values":   [float(v) for v in np.asarray(env_eval.partner_z_values)],
        "round_success_overall": round_success_overall,
        "n_rounds_total":     int(n_rounds_total),
        "per_z_success":      per_z_success.tolist(),
        "per_z_return":       per_z_return.tolist(),
        "per_z_rounds":       per_z_rounds.tolist(),
        "per_round_idx_success": per_round_success.tolist(),
        "per_round_idx_n":       per_round_n.tolist(),
        "mean_ep_return":     float(np.asarray(ep_return).mean()),
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path="config",
    config_name="ippo_rnn_coordination_grid",
)
def main(config):
    config = OmegaConf.to_container(config)
    num_seeds = config["NUM_SEEDS"]
    start_time = datetime.now()

    wandb.init(
        entity=config.get("ENTITY", ""),
        project=config.get("PROJECT", "coordination_grid"),
        tags=["IPPO", "RNN", "CoordinationGrid"],
        config=config,
        mode=config.get("WANDB_MODE", "disabled"),
        name=(
            f"ippo_rnn_coordination_grid"
            f"_R{config['ENV_KWARGS'].get('rounds_per_episode', 1)}"
            f"_zpool={config['ENV_KWARGS'].get('partner_z_values', [config['ENV_KWARGS'].get('partner_z', 0.5)])}"
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
        # Vmapped over seeds so shape is (num_seeds, num_updates).
        for k in ("round_success_rate", "partner_ep_success_rate",
                  "ep_return_mean", "ep_length_mean",
                  "reward_mean", "actor_loss", "value_loss",
                  "entropy", "entropy_t0", "entropy_tge1",
                  "t0_msg_none", "t0_msg_m0", "t0_msg_m1"):
            if k in m:
                v = np.asarray(m[k])  # (S, U)
                seq = v.mean(axis=0)  # avg over seeds
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

    # -------------- Held-out eval on val / test layout pools --------------
    eval_dirs: Dict[str, str] = config.get("EVAL_LAYOUTS_DIRS", {}) or {}
    n_eps = int(
        config.get("EVAL_EPISODES_PER_Z",
                   config.get("EVAL_TRIALS_PER_LAYOUT", 64))
    )
    if eval_dirs:
        params_seed0 = jax.tree_util.tree_map(
            lambda x: x[0], out["runner_state"][0].params
        )
        rng_eval = jax.random.PRNGKey(int(config.get("EVAL_SEED", 12345)))
        eval_summary: Dict[str, dict] = {}
        for split, ldir in eval_dirs.items():
            rng_eval, sub = jax.random.split(rng_eval)
            print(
                f"[eval] {split}: layouts_dir={ldir}, "
                f"{n_eps} episodes/z × {len(config['ENV_KWARGS'].get('partner_z_values', [config['ENV_KWARGS'].get('partner_z', 0.5)]))} z-values...",
                flush=True,
            )
            r = evaluate_policy(
                params_seed0, config, ldir, sub,
                n_episodes_per_z=n_eps,
            )
            eval_summary[split] = r
            print(
                f"[eval] {split}: round_success_overall={r['round_success_overall']:.3f}"
                f"  (n_rounds={r['n_rounds_total']})",
                flush=True,
            )
            print(f"[eval] {split}: per-z round-success:", flush=True)
            for zv, sv, nrv in zip(
                r["partner_z_values"], r["per_z_success"], r["per_z_rounds"]
            ):
                print(f"    z={zv:.2f}  succ={sv:.3f}  n_rounds={nrv}", flush=True)
            print(f"[eval] {split}: per-round-index success (across rounds):", flush=True)
            for ri, (sv, nv) in enumerate(
                zip(r["per_round_idx_success"], r["per_round_idx_n"])
            ):
                print(f"    round={ri:>2d}  succ={sv:.3f}  n={nv}", flush=True)
            wandb.log({
                f"eval_{split}/round_success_overall": r["round_success_overall"],
                **{f"eval_{split}/per_z_success/z={zv:.2f}": sv
                   for zv, sv in zip(r["partner_z_values"], r["per_z_success"])},
                **{f"eval_{split}/per_round/{ri}": sv
                   for ri, sv in enumerate(r["per_round_idx_success"])},
            })
        # Persist the full per-slot numbers if a save path was given.
        if save_path:
            eval_json_path = os.path.splitext(save_path)[0] + "_eval.json"
            import json as _json
            with open(eval_json_path, "w") as f:
                _json.dump({
                    k: {**v}
                    for k, v in eval_summary.items()
                }, f, indent=2)
            print(f"[eval] wrote per-z / per-round-index numbers -> "
                  f"{eval_json_path}", flush=True)

    return out


if __name__ == "__main__":
    main()
