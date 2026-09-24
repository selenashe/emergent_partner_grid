""" 
Based on PureJaxRL Implementation of PPO
"""

import jax
# jax.config.update("jax_disable_jit", True)
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Callable, Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked_v2 import overcooked_v2_layouts
# from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer
# from jaxmarl.viz.overcooked_visualizer_v2 import OvercookedVisualizer
from overcooked_visualizer_v2 import OvercookedVisualizer

import hydra
from omegaconf import OmegaConf
from datetime import datetime
import os
import wandb
import functools
import math
from typing import NamedTuple
from scipy.spatial.distance import jensenshannon

import matplotlib.pyplot as plt
from ippo_ff_overcooked_v2 import ActorCritic as ActorCritic_ff

def crop_local_view(obs: jnp.ndarray, radius: int = 1, pad_val: float = 0.0):
    """Return an egocentric (2r+1)x(2r+1) crop around the agent.
       Areas outside the map are padded with `pad_val` (usually 0)."""
    radius = 0
    h, w, c = obs.shape

    # agent is the only pixel with 1 in channel 0
    y, x = jnp.divmod(jnp.argmax(obs[..., 0].reshape(-1)), w)

    pad = radius
    obs_padded = jnp.pad(obs, ((pad, pad), (pad, pad), (0, 0)),
                         constant_values=pad_val)
    y, x = y + pad, x + pad
    return jax.lax.dynamic_slice(obs_padded,
                                 (y - radius, x - radius, 0),
                                 (2 * radius + 1, 2 * radius + 1, c))


def save_checkpoint(train_state, model_name="model-1"):
    import os, shutil
    from orbax.checkpoint import PyTreeCheckpointer

    ckpt_dir = os.path.join("/scratch/gpfs/rm4057", model_name)
    
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    if isinstance(train_state, dict):
        ckpt = {agent_id: state.params for agent_id, state in train_state.items()}
    else:
        ckpt = train_state.params

    PyTreeCheckpointer().save(ckpt_dir, ckpt)
    print(f"Checkpoint saved to {ckpt_dir}.")
    return


def load_checkpoint(model_name, ckpt_dir="/scratch/gpfs/rm4057"):
    import os
    from orbax.checkpoint import PyTreeCheckpointer

    model_dir = os.path.join(ckpt_dir, model_name)
    if os.path.exists(model_dir):
        print(f"Checkpoint found locally at {model_dir}.")
        return PyTreeCheckpointer().restore(model_dir)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {model_dir}.")


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
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
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class CNN(nn.Module):
    output_size: int = 64
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x, train=False):
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=8,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
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


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        embedding = obs

        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embed_model = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
        )
        embedding = jax.vmap(embed_model)(embedding)

        embedding = nn.LayerNorm()(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(x)

        actor_mean = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(embedding)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray



def calculate_jsd(agent_0_counts, agent_1_counts):
    agent_0_total = sum(agent_0_counts)
    agent_1_total = sum(agent_1_counts)

    if agent_0_total == 0 or agent_1_total == 0:
        return 0.0

    agent_0_probs = np.array(agent_0_counts) / agent_0_total
    agent_1_probs = np.array(agent_1_counts) / agent_1_total

    jsd = jensenshannon(agent_0_probs, agent_1_probs, base=2)
    return jsd



######################


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    randomness_range = 1

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env, replace_info=False)

    def create_learning_rate_fn():
        base_learning_rate = config["LR"]

        lr_warmup = config["LR_WARMUP"]
        update_steps = config["NUM_UPDATES"]
        warmup_steps = int(lr_warmup * update_steps)

        steps_per_epoch = config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]

        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=base_learning_rate,
            transition_steps=warmup_steps * steps_per_epoch,
        )
        cosine_epochs = max(update_steps - warmup_steps, 1)

        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
        )
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_steps * steps_per_epoch],
        )
        return schedule_fn

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0, end_value=0.0, transition_steps=config["REW_SHAPING_HORIZON"]
    )

    reward_shaping_factor = config['REW_SHAPING_FACTOR']

    def train(rng, preloaded_params=None, preloaded_params_2=None, model1_params=None, model2_params=None):

        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env.agents[0]).n + 1, config=config)
        num_actions = env.action_space(env.agents[0]).n

        # network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)

        rng, _rng = jax.random.split(rng)
        # init_x = (
        #     jnp.zeros((1, config["NUM_ENVS"], *env.observation_space().shape)),
        #     jnp.zeros((1, config["NUM_ENVS"])),
        # )

        # crop_local_view
        init_x = (
            jnp.zeros((1, config["NUM_ENVS"], 1, 1, env.observation_space().shape[-1])),
            jnp.zeros((1, config["NUM_ENVS"])),
        )

        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )

        network_params = network.init(_rng, init_hstate, init_x)

        # OPTIONAL: INIT ANOTHER NETWORK
        if preloaded_params:
            network2 = ActorCritic_ff(env.action_space().n, activation="tanh")       
            network2_params = preloaded_params
            network2_params_list = [preloaded_params, preloaded_params_2]

            rng2, _rng2 = jax.random.split(rng)
            init_x2 = jnp.zeros(env.observation_space().shape)
            init_x2 = init_x2.flatten()            
            network2.init(_rng2, init_x2)

        # EXPERIMENT 1
        if model1_params and model2_params:
            network3 = ActorCritic_ff(env.action_space().n, activation="tanh")
            rng2, _rng2 = jax.random.split(rng)
            init_x3 = jnp.zeros(env.observation_space().shape)
            init_x3 = init_x3.flatten()            
            network3.init(_rng2, init_x3)



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
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        if preloaded_params:
            train_state2 = TrainState.create(
                apply_fn=network2.apply,
                params=network2_params,
                tx=tx,
            )


        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    update_step,
                    hstate,
                    rng,
                    model_status_1, 
                    prob_0,
                    prob_1,
                ) = runner_state



                # rng, rng_slow0, rng_slow1, rng_slow2, _rng = jax.random.split(rng, 5)
                # rng, rng_prob0, rng_prob1, _rng = jax.random.split(rng, 4)
                rng, rng_low, rng_high, rng_choice, _rng = jax.random.split(rng, 5)
                last_done_agent1 = last_done.reshape(len(env.agents), config["NUM_ENVS"])[1]

                # low_prob = jax.random.uniform(rng_low, (config["NUM_ENVS"],)) * randomness_range
                # high_prob = randomness_range + jax.random.uniform(rng_high, (config["NUM_ENVS"],)) * randomness_range
                # low_prob = jax.random.uniform(rng_low, (config["NUM_ENVS"],)) * randomness_range
                # high_prob = jax.random.uniform(rng_high, (config["NUM_ENVS"],)) * randomness_range

                low_prob  = jax.random.uniform(rng_low,  (config["NUM_ENVS"],)) * 0.1
                high_prob = 0.9 + jax.random.uniform(rng_high, (config["NUM_ENVS"],)) * 0.1

                # low_prob = jax.random.choice(rng_low, jnp.array([0.0, 0.05, 0.1]), (config["NUM_ENVS"],))
                # high_prob = jax.random.choice(rng_high, jnp.array([0.9, 0.95, 1.0]), (config["NUM_ENVS"],))

                # low_prob  = jnp.full((config["NUM_ENVS"],), 0)
                # high_prob = jnp.full((config["NUM_ENVS"],), 0)


                mask = jax.random.bernoulli(rng_choice, 0.5, (config["NUM_ENVS"],))
                new_prob_0 = jnp.where(mask, low_prob, high_prob)
                new_prob_1 = jnp.where(mask, high_prob, low_prob)


                prob_0 = jnp.where(last_done_agent1, new_prob_0, prob_0)
                prob_1 = jnp.where(last_done_agent1, new_prob_1, prob_1)


         
                # obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                #     -1, *env.observation_space().shape
                # )

                obs_stack = jnp.stack([jax.vmap(crop_local_view)(last_obs[a]) for a in env.agents])
                obs_batch = obs_stack.reshape(-1, *obs_stack.shape[2:])      


                ac_in = (
                    obs_batch[jnp.newaxis, :],
                    last_done[jnp.newaxis, :],
                )

                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)

                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )

                env_act = {k: v.flatten() for k, v in env_act.items()}

                # NETWORK 2
                if preloaded_params:
                    obs_batch_ff = batchify(last_obs, env.agents, config["NUM_ACTORS"])

                    current_network2_params = preloaded_params

                    pi2, value2 = network2.apply(current_network2_params, obs_batch_ff)
                    action2 = pi2.sample(seed=_rng)
                    env_act2 = unbatchify(action2, env.agents, config["NUM_ENVS"], env.num_agents)
                    env_act2 = {k:v.flatten() for k,v in env_act2.items()}

                    rand_val = jax.random.uniform(_rng, (config["NUM_ENVS"],))
                    random_action = jax.random.randint(_rng, (config["NUM_ENVS"],), 0, num_actions)

                    agent1_action = env_act2["agent_1"]
                    agent1_action = jnp.where(rand_val < prob_0, agent1_action, random_action)


                    env_act = {"agent_0": env_act["agent_0"],
                               "agent_1": agent1_action,
                            }                



                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                original_reward = jnp.array([reward[a] for a in env.agents])

                current_timestep = (
                    update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                )
                anneal_factor = rew_shaping_anneal(current_timestep)*reward_shaping_factor
                reward = jax.tree_util.tree_map(
                    lambda x, y: x + y * anneal_factor, reward, info["shaped_reward"]
                )

                shaped_reward = jnp.array(
                    [info["shaped_reward"][a] for a in env.agents]
                )
                combined_reward = jnp.array([reward[a] for a in env.agents])

                info["shaped_reward"] = shaped_reward
                info["original_reward"] = original_reward
                info["anneal_factor"] = jnp.full_like(shaped_reward, anneal_factor)
                info["combined_reward"] = combined_reward

                # info = jax.tree_util.tree_map(
                #     lambda x: x.reshape((config["NUM_ACTORS"])), info
                # )
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done_batch,
                    update_step,
                    hstate,
                    rng,
                    model_status_1,
                    prob_0,
                    prob_1,
                )
                return runner_state, transition

            initial_hstate = runner_state[-5]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, update_step, hstate, rng, model_status_1, prob_0, prob_1 = (
                runner_state
            )

            # last_obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
            #     -1, *env.observation_space().shape
            # )

            last_stack = jnp.stack([jax.vmap(crop_local_view)(last_obs[a]) for a in env.agents])
            last_obs_batch = last_stack.reshape(-1, *last_stack.shape[2:])

            ac_in = (
                last_obs_batch[jnp.newaxis, :],
                last_done[jnp.newaxis, :],
            )
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
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

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                @jax.jit
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    @jax.jit
                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done),
                        )

                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
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
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = (
                    update_state
                )
                rng, _rng = jax.random.split(rng)
                # batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                # assert (
                #     batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                # ), "batch size must be equal to number of steps * number of actors"
                # permutation = jax.random.permutation(_rng, batch_size)
                # batch = (init_hstate, traj_batch, advantages, targets)
                # batch = jax.tree_util.tree_map(
                #     lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                # )
                # shuffled_batch = jax.tree_util.tree_map(
                #     lambda x: jnp.take(x, permutation, axis=0), batch
                # )
                # minibatches = jax.tree_util.tree_map(
                #     lambda x: jnp.reshape(
                #         x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                #     ),
                #     shuffled_batch,
                # )

                init_hstate = jnp.reshape(init_hstate, (1, config["NUM_ACTORS"], -1))
                batch = (
                    init_hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
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
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate.squeeze(),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            def callback(metric):
                wandb.log(metric)

            update_step = update_step + 1
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            jax.debug.callback(callback, metric)

            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                hstate,
                rng,
                model_status_1, 
                prob_0,
                prob_1,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        
        model_status_init_1 = jnp.zeros(config["NUM_ENVS"], dtype=bool)


        _rng, rng_low, rng_high, rng_choice = jax.random.split(_rng, 4)
        # low_init = jax.random.uniform(rng_low, (config["NUM_ENVS"],)) * randomness_range
        # high_init = jax.random.uniform(rng_high, (config["NUM_ENVS"],)) * randomness_range

        low_init  = jax.random.uniform(rng_low, (config["NUM_ENVS"],)) * 0.1
        high_init = 0.9 + jax.random.uniform(rng_high, (config["NUM_ENVS"],)) * 0.1

        # low_init = jax.random.choice(rng_low, jnp.array([0.0, 0.05, 0.1]), (config["NUM_ENVS"],))
        # high_init = jax.random.choice(rng_high, jnp.array([0.9, 0.95, 1.0]), (config["NUM_ENVS"],))

        low_init  = jnp.full((config["NUM_ENVS"],), 0)
        high_init = jnp.full((config["NUM_ENVS"],), 0)

        mask_init = jax.random.bernoulli(rng_choice, 0.5, (config["NUM_ENVS"],))
        prob_init_0 = jnp.where(mask_init, low_init, high_init)
        prob_init_1 = jnp.where(mask_init, high_init, low_init)


        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            0,
            init_hstate,
            _rng,
            model_status_init_1,
            prob_init_0,
            prob_init_1,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


@hydra.main(
    version_base=None, config_path="config", config_name="ippo_rnn_overcooked_v2"
)
def main(config):
    config = OmegaConf.to_container(config)

    layout_name = config["ENV_KWARGS"]["layout"]
    config["ENV_KWARGS"]["layout"] = overcooked_v2_layouts[layout_name]    
    num_seeds = config["NUM_SEEDS"]

    start_time = datetime.now()

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN", "OvercookedV2"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"ippo_rnn_overcooked_v2_{layout_name}",
    )

    ckpt_dir="/scratch/gpfs/rm4057"
    # ckpt_dir = "/checkpoints"

    if config['LOAD_MODEL']:
        model_name = config['LOAD_MODEL_NAME']
        loaded_ckpt = load_checkpoint(model_name)
    else: loaded_ckpt = None

    if config['LOAD_MODEL_2']:
        model_name = config['LOAD_MODEL_2_NAME']
        loaded_ckpt_2 = load_checkpoint(model_name)
    else: loaded_ckpt_2 = None

    if config['RNN_LOAD_MODEL']:
        model_name = config['RNN_LOAD_MODEL_NAME']
        loaded_ckpt_RNN = load_checkpoint(model_name)
        train_state = None
    else: loaded_ckpt_RNN = None

    if config['EXPERIMENT_1']:
        model1_params = load_checkpoint(config['EXP1_MODEL_1'], ckpt_dir)
        model2_params = load_checkpoint(config['EXP1_MODEL_2'], ckpt_dir)
    else: model1_params, model2_params = None, None


    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, num_seeds)
        train_jit = jax.jit(make_train(config))
        # train_jit = (make_train(config))
        # out = jax.vmap(train_jit)(rngs)
        
        out = jax.vmap(lambda rng: train_jit(rng, preloaded_params=loaded_ckpt, preloaded_params_2=loaded_ckpt_2,
                                             model1_params=model1_params, model2_params=model2_params))(rngs)


    clock1 = datetime.now() - start_time
    wandb.log({"clocktime1": clock1.total_seconds()})

    train_state = jax.tree_map(lambda x: x[0], out["runner_state"][0])

    # print("** Saving Results **")
    # filename = f'{config["ENV_NAME"]}_{layout_name}_seed{config["SEED"]}'
    # state_seq, reward_seq = get_rollout(train_state, config, loaded_ckpt, loaded_ckpt_RNN, model1_params=model1_params, model2_params=model2_params)
    # viz = OvercookedVisualizer()
    # viz.animate(state_seq, reward_seq, agent_view_size=5, filename=f"{filename}.mp4")
    # wandb.log({"animation": wandb.Video(f"{filename}.mp4", format="mp4")})

    # clock2 = datetime.now() - start_time
    # wandb.log({"clocktime2": clock2.total_seconds()})
    # wandb.log({"notes": "reward horizon 0e6"})

    if config['SAVE_MODEL']:
        model_name = config['SAVE_MODEL_NAME']
        save_checkpoint(train_state, model_name)

if __name__ == "__main__":
    main()