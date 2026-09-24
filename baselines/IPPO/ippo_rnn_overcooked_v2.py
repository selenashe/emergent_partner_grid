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

def get_rollout(train_state, config, network2_params=None, rnn_params=None, model1_params=None, model2_params=None):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    # network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
    network = ActorCriticRNN(env.action_space(env.agents[0]).n + 1, config=config)
    network2 = ActorCritic_ff(env.action_space().n, activation="tanh")
    network3 = ActorCritic_ff(env.action_space().n, activation="tanh")

    # rng, _rng = jax.random.split(rng)

    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    init_x = (
        jnp.zeros(
            (1, 1, math.prod(env.observation_space().shape))
        ),
        jnp.zeros((1, 1)),
    )
    hstate_0 = ScannedRNN.initialize_carry(1, config["GRU_HIDDEN_DIM"])
    hstate_1 = ScannedRNN.initialize_carry(1, config["GRU_HIDDEN_DIM"])
    # network_params = network.init(key, init_hstate, init_x)

    if rnn_params:
        network_params = rnn_params
    else:
        network_params = train_state.params

    done = False
    obs, state = env.reset(key_r)
    state_seq = [state]
    obs_save_0 = np.array(obs['agent_0'])
    obs_seq_0 = [obs_save_0]
    obs_save_1 = np.array(obs['agent_1'])
    obs_seq_1 = [obs_save_1]

    action_seq_0 = []
    action_seq_1 = []

    last_done = jnp.zeros((1), dtype=bool)

    rewards = []
    shaped_rewards = [0]
    reward_seq = [0]
    high_level_action_sums = None
    model_status_1 = 0

    if network2_params:
         init_x = jnp.zeros(env.observation_space().shape)
         init_x = init_x.flatten()
         network2.init(key_a, init_x)   


    timestep = 0
    agent0_last_action = 0
    agent0_action_counter = 0
    agent1_last_action = 0
    agent1_action_counter = 0
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        obs_flattened_0 = obs_save_0.flatten().reshape(1, *env.observation_space().shape)
        obs_flattened_1 = obs_save_1.flatten().reshape(1, *env.observation_space().shape)

        ac_in_0 = (
            obs_flattened_0[np.newaxis, :],
            last_done[np.newaxis, :],
        )

        ac_in_1 = (
            obs_flattened_1[np.newaxis, :],
            last_done[np.newaxis, :],
        )

        hstate_0, pi_0, value_0 = network.apply(network_params, hstate_0, ac_in_0)
        hstate_1, pi_1, value_1 = network.apply(network_params, hstate_1, ac_in_1)

        
        AGENT_0_COOLDOWN = 2  # e.g. act every 3 steps
        WAIT_ACTION = 4       # environment’s wait action

        if agent0_action_counter == 0:
            agent0_last_action = pi_0.sample(seed=key_a0)[0][0]
            agent0_action_counter = AGENT_0_COOLDOWN
            action_0 = agent0_last_action
        else:
            action_0 = WAIT_ACTION
            agent0_action_counter -= 1


        action_1 = pi_1.sample(seed=key_a0)[0][0]

        if network2_params:
            obs_ff = {k: v.flatten() for k, v in obs.items()}

            pi_1, _ = network2.apply(network2_params, obs_ff["agent_1"])
            action_1 = pi_1.sample(seed=key_a1)

        actions = {
            "agent_0": action_0,  # Convert to scalar
            "agent_1": action_1   # Convert to scalar
        }

        if model1_params and model2_params:
            obs_ff = {k: v.flatten() for k, v in obs.items()}       
            a0 = pi_0.sample(seed=key_a0)[0][0]
            switch = a0 == 6
            if switch:
                model_status_1 = 1 - model_status_1

            agent0_action = 4 if switch else a0
            pi_model2_1, _ = network3.apply(model1_params, obs_ff["agent_1"])
            pi_model2_2, _ = network3.apply(model2_params, obs_ff["agent_1"])


            ##
            # speeds 6, 1 does NOT work
            AGENT_1_1_SPEED = 5
            AGENT_1_2_SPEED = 1
            agent1_last_action = 4
            if model_status_1 == 0:
                if agent1_action_counter >= AGENT_1_1_SPEED:
                    agent1_last_action = pi_model2_1.sample(seed=key_a1)
                    agent1_action_counter = 0
            else:
                if agent1_action_counter >= AGENT_1_2_SPEED:
                    agent1_last_action = pi_model2_2.sample(seed=key_a1)
                    agent1_action_counter = 0

            agent1_action = agent1_last_action
            agent1_action_counter += 1
            ##

            actions = {"agent_0": agent0_action, "agent_1": agent1_action}


        obs, state, reward, done, info = env.step(key_s, state, actions)
        timestep += 1
        done = done["__all__"]

        state_seq.append(state)
        reward_seq.append(reward)

        obs_save_0 = np.array(obs['agent_0'])
        obs_save_1 = np.array(obs['agent_1'])

        obs_seq_0.append(obs_save_0)
        action_seq_0.append(actions['agent_0'])
        obs_seq_1.append(obs_save_1)
        action_seq_1.append(actions['agent_1'])

        ####################################################

        if high_level_action_sums is None:
            high_level_action_sums = {
                agent: {action: 0 for action in info["high_level_actions"][agent]}
                for agent in info["high_level_actions"]
            }

        for agent in info["high_level_actions"]:
            for action, value in info["high_level_actions"][agent].items():
                high_level_action_sums[agent][action] += int(value)

        rewards.append(reward['agent_0'])
        shaped_rewards.append(info["shaped_reward"]['agent_0'])

    rollout_reward = sum(rewards)
    print(f'rollout_reward {rollout_reward}')
    wandb.log({"rollout_reward": rollout_reward})

    agent_0_counts = list(high_level_action_sums["agent_0"].values())
    agent_1_counts = list(high_level_action_sums["agent_1"].values())
    jsd = calculate_jsd(agent_0_counts, agent_1_counts)
    wandb.log({"Jensen-Shannon Divergence": jsd})
    wandb.log({"high_level_action_sums": high_level_action_sums})

    from matplotlib import pyplot as plt

    # Plot the distributions
    labels = high_level_action_sums["agent_0"].keys()
    x = np.arange(len(labels))  # label locations
    width = 0.35  # width of the bars

    fig, ax = plt.subplots(figsize=(10, 10))
    rects1 = ax.bar(x - width/2, agent_0_counts, width, label='Agent 0', color='red')
    rects2 = ax.bar(x + width/2, agent_1_counts, width, label='Agent 1', color='blue')

    # Add labels, title, and legend
    ax.set_ylabel('Counts')
    ax.set_title('High-Level Action Counts by Agent')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()

    # Log the plot to wandb
    wandb.log({"high_level_action_distribution": wandb.Image(fig)})

    ####

    cumulative_rewards = np.cumsum(rewards)
    serve_timesteps = [i for i, r in enumerate(rewards) if r > 0]

    # Step 2: Create corresponding cumulative reward values (1, 2, 3, ...) at those steps
    if len(serve_timesteps) > 1:

        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(range(len(cumulative_rewards)), cumulative_rewards)
        ax3.set_title("Cumulative Reward Over Time")
        ax3.set_xlabel("Timestep")
        ax3.set_ylabel("Cumulative Reward")

        # wandb.log({"cumulative_reward_over_time": wandb.Image(fig3)})

        wandb.log({
            "cumulative_reward_over_time": wandb.Image(fig3),
        })


    return state_seq, reward_seq


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

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
        # network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)

        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, config["NUM_ENVS"], *env.observation_space().shape)),
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
                    wait_buffer_0,
                    wait_buffer_1,
                    slow_delay_0,
                    slow_delay_1,
                ) = runner_state



                rng, rng_slow0, rng_slow1, rng_slow2, _rng = jax.random.split(rng, 5)
                allowed_speeds = jnp.array([1, 2, 3, 4, 7, 8, 9])
                low_set = jnp.array([1,2,3])
                high_set = jnp.array([4,7,8,9])
                new_slow_delay_0 = jax.random.choice(rng_slow0, allowed_speeds, (config["NUM_ENVS"],))
                idx_low = jax.random.randint(rng_slow1, (config["NUM_ENVS"],), 0, 3)
                idx_high = jax.random.randint(rng_slow2, (config["NUM_ENVS"],), 0, 4)                
                new_slow_delay_1 = jnp.where(jnp.isin(new_slow_delay_0, low_set), high_set[idx_high], low_set[idx_low])

                last_done_agent1 = last_done.reshape(len(env.agents), config["NUM_ENVS"])[1]
                slow_delay_0 = jnp.where(last_done_agent1, new_slow_delay_0, slow_delay_0)
                slow_delay_1 = jnp.where(last_done_agent1, new_slow_delay_1, slow_delay_1)

                new_wait_buffer_0 = jax.random.randint(rng_slow0, (config["NUM_ENVS"],), 0, 3)
                current_slow_delay = jnp.where(model_status_1, slow_delay_1, slow_delay_0)
                new_wait_buffer_1 = jax.random.randint(rng_slow1, (config["NUM_ENVS"],), 0, current_slow_delay + 1)
                
                wait_buffer_0 = jnp.where(last_done_agent1, new_wait_buffer_0, wait_buffer_0)
                wait_buffer_1 = jnp.where(last_done_agent1, new_wait_buffer_1, wait_buffer_1)


                obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                    -1, *env.observation_space().shape
                )

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

                    switch_interval = jnp.array(config["NETWORK2_SWITCH_INTERVAL"])
                    if preloaded_params_2 is None:
                        current_network2_params = preloaded_params
                    else:
                        network2_params_list = [preloaded_params, preloaded_params_2]
                        idx = jnp.mod(jnp.floor_divide(update_step, switch_interval), 2)
                        current_network2_params = jax.lax.switch(
                            idx,
                            [lambda _, i=i: network2_params_list[i] for i in range(2)],
                            None
                        )


                    pi2, value2 = network2.apply(current_network2_params, obs_batch_ff)
                    action2 = pi2.sample(seed=_rng)
                    log_prob2 = pi2.log_prob(action2)
                    env_act2 = unbatchify(action2, env.agents, config["NUM_ENVS"], env.num_agents)
                    env_act2 = {k:v.flatten() for k,v in env_act2.items()}

                    env_act = {"agent_0": env_act["agent_0"],
                            "agent_1": env_act2["agent_1"],
                            }                


                if model1_params and model2_params:
                    obs_batch_ff = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                    WAIT_ACTION = 4

                    action1 = env_act['agent_0']

                    # # Wait logic
                    not_waiting_0 = wait_buffer_0 == 0
                    action1 = jnp.where(wait_buffer_0 > 0, WAIT_ACTION, action1)
                    COOLDOWN_0 = 2
                    wait_buffer_0 = jnp.where(not_waiting_0, COOLDOWN_0, wait_buffer_0)
                    wait_buffer_0 = jnp.maximum(0, wait_buffer_0 - 1)

                    obs_reshaped = obs_batch_ff.reshape((len(env.agents), config["NUM_ENVS"], -1))

                    pi_model2_1, _ = network3.apply(model1_params, obs_batch_ff)  # (16, obs_dim) -> (16,) actions
                    pi_model2_2, _ = network3.apply(model2_params, obs_batch_ff)
                    a2_1_both_agents = pi_model2_1.sample(seed=_rng)
                    a2_2_both_agents = pi_model2_2.sample(seed=_rng)

                    env_act_a2_1 = unbatchify(a2_1_both_agents, env.agents, config["NUM_ENVS"], env.num_agents)
                    env_act_a2_2 = unbatchify(a2_2_both_agents, env.agents, config["NUM_ENVS"], env.num_agents)
                    env_act_a2_1 = {k:v.flatten() for k,v in env_act_a2_1.items()}
                    env_act_a2_2 = {k:v.flatten() for k,v in env_act_a2_2.items()}

                    a2_1_agent_1 = env_act_a2_1["agent_1"]
                    a2_2_agent_1 = env_act_a2_2["agent_1"]

                    switch_mask = jnp.equal(action1, 6)
                    model_status_1 = jnp.where(switch_mask, jnp.logical_not(model_status_1), model_status_1)

                    agent1_action = jnp.where(switch_mask, 4, action1)
                    agent2_action = jnp.where(model_status_1, a2_1_agent_1, a2_2_agent_1)

                    # Wait logic - it may be correct
                    not_waiting = wait_buffer_1 == 0
                    agent2_action = jnp.where(wait_buffer_1 > 0, WAIT_ACTION, agent2_action)
                    new_delay = jnp.where(model_status_1, slow_delay_1, slow_delay_0)
                    # wait_buffer_1 = jnp.where(not_waiting, new_delay, wait_buffer_1)
                    wait_buffer_1 = jnp.where(switch_mask, new_delay, jnp.where(not_waiting, new_delay, wait_buffer_1))
                    wait_buffer_1 = jnp.maximum(0, wait_buffer_1 - 1)

                    env_act = {
                        "agent_0": agent1_action,
                        "agent_1": agent2_action,
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
                    wait_buffer_0,
                    wait_buffer_1,
                    slow_delay_0,
                    slow_delay_1,
                )
                return runner_state, transition

            initial_hstate = runner_state[-7]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, update_step, hstate, rng, model_status_1, wait_buffer_0, wait_buffer_1, slow_delay_0, slow_delay_1 = (
                runner_state
            )
            # last_obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
            #     -1, *env.observation_space().shape
            # )
            # last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            last_obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                -1, *env.observation_space().shape
            )
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
                wait_buffer_0,
                wait_buffer_1,
                slow_delay_0,
                slow_delay_1,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        model_status_init_1 = jnp.zeros(config["NUM_ENVS"], dtype=bool)
        wait_buffer_init_0 = jnp.zeros(config["NUM_ENVS"], dtype=jnp.int32)        
        wait_buffer_init_1 = jnp.zeros(config["NUM_ENVS"], dtype=jnp.int32)        

        allowed_speeds = jnp.array([1, 2, 3, 4, 7, 8, 9])
        low_set = jnp.array([1,2,3])
        high_set = jnp.array([4,7,8,9])
        _rng, rng_init0, rng_init1, rng_init2 = jax.random.split(_rng, 4)
        slow_delay_init_0 = jax.random.choice(rng_init0, allowed_speeds, (config["NUM_ENVS"],))
        idx_low_init = jax.random.randint(rng_init1, (config["NUM_ENVS"],), 0, 3)
        idx_high_init = jax.random.randint(rng_init2, (config["NUM_ENVS"],), 0, 4)
        slow_delay_init_1 = jnp.where(jnp.isin(slow_delay_init_0, low_set), high_set[idx_high_init], low_set[idx_low_init])


        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            0,
            init_hstate,
            _rng,
            model_status_init_1,
            wait_buffer_init_0,
            wait_buffer_init_1,
            slow_delay_init_0,
            slow_delay_init_1,
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

    clock2 = datetime.now() - start_time
    wandb.log({"clocktime2": clock2.total_seconds()})
    wandb.log({"notes": "reward horizon 0e6"})

    if config['SAVE_MODEL']:
        model_name = config['SAVE_MODEL_NAME']
        save_checkpoint(train_state, model_name)

if __name__ == "__main__":
    main()