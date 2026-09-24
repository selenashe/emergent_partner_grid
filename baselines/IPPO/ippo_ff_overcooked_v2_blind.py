""" 
Based on PureJaxRL Implementation of PPO
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked_v2 import overcooked_v2_layouts
# from jaxmarl.viz.overcooked_visualizer_v2 import OvercookedVisualizer
from overcooked_visualizer_v2 import OvercookedVisualizer
import hydra
from omegaconf import OmegaConf
import wandb
from scipy.spatial.distance import jensenshannon

import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import orbax.checkpoint as orbax
import shutil
import os
# os.environ["WANDB_DIR"] = "/scratch/gpfs/rm4057/wandb_logs"
os.environ["WANDB_DIR"] = os.path.expanduser("~/wandb2")
import uuid
import random


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


save_schedule = [10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
_next_ckpt = 0

def _checkpoint_if_due(state, step):
    global _next_ckpt
    step = int(jax.device_get(step))          # turn JAX scalar into Python int
    if _next_ckpt < len(save_schedule) and step >= save_schedule[_next_ckpt]:
        save_checkpoint(state, f"model_step_{step}")
        _next_ckpt += 1


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

    # === ADD THE FOLLOWING LINES ===
    artifact = wandb.Artifact(model_name, type="model")
    artifact.add_dir(ckpt_dir)
    wandb.log_artifact(artifact)

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


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:

            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
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


def calculate_jsd(agent_0_counts, agent_1_counts):
    agent_0_probs = np.array(agent_0_counts) / sum(agent_0_counts)
    agent_1_probs = np.array(agent_1_counts) / sum(agent_1_counts)

    jsd = jensenshannon(agent_0_probs, agent_1_probs, base=2)
    return jsd

def get_rollout(train_state, config, model1_params=None, model2_params=None):

    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
    network2 = ActorCritic(env.action_space().n, activation="tanh")
    network3 = ActorCritic(env.action_space().n, activation="tanh")

    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    init_x = jnp.zeros(env.observation_space().shape)
    init_x = init_x.flatten()

    network.init(key_a, init_x)

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


    timestep = 0
    agent0_last_action = 0
    agent0_action_counter = 0
    agent1_last_action = 0
    agent1_action_counter = 0

    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)


        obs = {k: v.flatten() for k, v in obs.items()}
        pi_0, _ = network.apply(network_params, obs["agent_0"])
        pi_1, _ = network.apply(network_params, obs["agent_1"])
        actions = {"agent_0": pi_0.sample(seed=key_a0), "agent_1": pi_1.sample(seed=key_a1)}

        
        AGENT_0_COOLDOWN = 2  # e.g. act every 3 steps
        WAIT_ACTION = 4       # environment’s wait action

        if agent0_action_counter == 0:
            agent0_last_action = pi_0.sample(seed=key_a0)
            agent0_action_counter = AGENT_0_COOLDOWN
            action_0 = agent0_last_action
        else:
            action_0 = WAIT_ACTION
            agent0_action_counter -= 1


        action_1 = actions['agent_1']

        actions = {
            "agent_0": action_0,  # Convert to scalar
            "agent_1": action_1   # Convert to scalar
        }

        if model1_params and model2_params:
            obs_ff = {k: v.flatten() for k, v in obs.items()}       
            a0 = pi_0.sample(seed=key_a0)
            switch = a0 == 6
            if switch:
                model_status_1 = 1 - model_status_1

            agent0_action = 4 if switch else a0
            pi_model2_1, _ = network3.apply(model1_params, obs_ff["agent_1"])
            pi_model2_2, _ = network3.apply(model2_params, obs_ff["agent_1"])


            AGENT_1_1_SPEED = 6
            AGENT_1_2_SPEED = 0
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
    env = jaxmarl.make(config["ENV_NAME"], 
                       **config["ENV_KWARGS"])

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"] 
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    
    env = LogWrapper(env, replace_info=False)
    
    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac


    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.,
        end_value=0.,
        transition_steps=config["REW_SHAPING_HORIZON"]
    )

    reward_shaping_factor = config['REW_SHAPING_FACTOR']

    def train(rng, preloaded_params=None, model1_params=None, model2_params=None):


        # INIT NETWORK
        network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
        num_actions = env.action_space(env.agents[0]).n
        rng, _rng = jax.random.split(rng)

        # init_x = jnp.zeros(env.observation_space().shape)
        # init_x = jnp.zeros((1, 1, env.observation_space().shape[-1]))
        # init_x = init_x.flatten()

        init_x = jnp.zeros((1, 1, env.observation_space().shape[-1])).flatten()

        network_params = network.init(_rng, init_x)


        if preloaded_params:
            network2 = ActorCritic(env.action_space().n, activation="tanh")       
            network2_params = preloaded_params
            rng2, _rng2 = jax.random.split(rng)
            init_x2 = jnp.zeros(env.observation_space().shape)
            init_x2 = init_x2.flatten()            
            network2.init(_rng2, init_x2)

            
        # EXPERIMENT 1
        if model1_params and model2_params:
            network3 = ActorCritic(env.action_space().n, activation="tanh")
            rng2, _rng2 = jax.random.split(rng)
            init_x3 = jnp.zeros(env.observation_space().shape)
            init_x3 = init_x3.flatten()            
            network3.init(_rng2, init_x3)


        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))

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
        
        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                # train_state, env_state, last_obs, update_step, rng = runner_state

                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    update_step,
                    rng,
                    model_status_1, 
                    prob_0,
                    prob_1,
                ) = runner_state

                # SELECT ACTION

                rng, rng_low, rng_high, rng_choice, _rng = jax.random.split(rng, 5)
                last_done_agent1 = last_done.reshape(len(env.agents), config["NUM_ENVS"])[1]

                low_prob  = jax.random.uniform(rng_low,  (config["NUM_ENVS"],)) * 0.1
                high_prob = 0.9 + jax.random.uniform(rng_high, (config["NUM_ENVS"],)) * 0.1

                mask = jax.random.bernoulli(rng_choice, 0.5, (config["NUM_ENVS"],))
                new_prob_0 = jnp.where(mask, low_prob, high_prob)
                new_prob_1 = jnp.where(mask, high_prob, low_prob)


                prob_0 = jnp.where(last_done_agent1, new_prob_0, prob_0)
                prob_1 = jnp.where(last_done_agent1, new_prob_1, prob_1)


                obs_batch1 = batchify(last_obs, env.agents, config["NUM_ACTORS"])



                obs_stack = jnp.stack([jax.vmap(crop_local_view)(last_obs[a]) for a in env.agents])
                obs_batch = obs_stack.reshape(obs_stack.shape[0] * obs_stack.shape[1], -1)
                # last_obs_batch = last_stack.reshape(last_stack.shape[0] * last_stack.shape[1], -1)


                pi, value = network.apply(train_state.params, obs_batch)

                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)

                env_act = {k:v.flatten() for k,v in env_act.items()}

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
                
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    rng_step, env_state, env_act
                )

                info["reward"] = reward["agent_0"]


                current_timestep = update_step*config["NUM_STEPS"]*config["NUM_ENVS"]
                reward = jax.tree_map(lambda x,y: x+y*rew_shaping_anneal(current_timestep)*reward_shaping_factor, reward, info["shaped_reward"])
                # reward = jax.tree_map(lambda x,y: x*0+y*rew_shaping_anneal(current_timestep), reward, info["shaped_reward"])

                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                )
                runner_state = (train_state, 
                                env_state, 
                                obsv,
                                done_batch, 
                                update_step, 
                                rng,
                                model_status_1,
                                prob_0,
                                prob_1,                             
                                )

                return runner_state, (transition, info)

            runner_state, (traj_batch, info) = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            
            # CALCULATE ADVANTAGE
            # train_state, env_state, last_obs, update_step, rng = runner_state
            train_state, env_state, last_obs, last_done, update_step, rng, model_status_1, prob_0, prob_1 = (
                runner_state
            )

            last_stack = jnp.stack([jax.vmap(crop_local_view)(last_obs[a]) for a in env.agents])
            last_obs_batch = last_stack.reshape(last_stack.shape[0] * last_stack.shape[1], -1)


            # last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            _, last_val = network.apply(train_state.params, last_obs_batch)

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
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
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
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = info
            current_timestep = update_step*config["NUM_STEPS"]*config["NUM_ENVS"]
            metric["shaped_reward"] = metric["shaped_reward"]["agent_0"]
            metric["shaped_reward_annealed"] = metric["shaped_reward"]*rew_shaping_anneal(current_timestep)
            
            rng = update_state[-1]

            def callback(metric):
                wandb.log(
                    metric
                )
            update_step = update_step + 1
            metric = jax.tree_map(lambda x: x.mean(), metric)
            metric["update_step"] = update_step
            metric["env_step"] = update_step*config["NUM_STEPS"]*config["NUM_ENVS"]
            jax.debug.callback(callback, metric)
            # jax.debug.callback(_checkpoint_if_due, train_state, metric["env_step"])


            # runner_state = (train_state, env_state, last_obs, update_step, rng)

            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                rng,
                model_status_1, 
                prob_0,
                prob_1,
            )

            return runner_state, metric

        rng, _rng = jax.random.split(rng)

        _rng, rng_low, rng_high, rng_choice = jax.random.split(_rng, 4)

        model_status_init_1 = jnp.zeros(config["NUM_ENVS"], dtype=bool)

        low_init  = jax.random.uniform(rng_low, (config["NUM_ENVS"],)) * 0.1
        high_init = 0.9 + jax.random.uniform(rng_high, (config["NUM_ENVS"],)) * 0.1

        mask_init = jax.random.bernoulli(rng_choice, 0.5, (config["NUM_ENVS"],))
        prob_init_0 = jnp.where(mask_init, low_init, high_init)
        prob_init_1 = jnp.where(mask_init, high_init, low_init)


        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            0,
            _rng,
            model_status_init_1,
            prob_init_0,
            prob_init_1,
        )

        # runner_state = (train_state, env_state, obsv, 0, _rng)

        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metric}

    return train



@hydra.main(version_base=None, config_path="config", config_name="ippo_ff_overcooked_v2")
def main(config):
    config = OmegaConf.to_container(config) 
    layout_name = config["ENV_KWARGS"]["layout"]
    config["ENV_KWARGS"]["layout"] = overcooked_v2_layouts[layout_name]

    # NUM_LAYOUTS = 1940
    # 194

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "FF"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f'ippo_ff_overcooked_{layout_name}'
    )

    rng = jax.random.PRNGKey(config["SEED"])
    ckpt_dir="/scratch/gpfs/rm4057"


    if config['LOAD_MODEL']:
        model_name = config['LOAD_MODEL_NAME']
        loaded_ckpt = load_checkpoint(model_name)
    else: loaded_ckpt = None


    if config['EXPERIMENT_1']:
        model1_params = load_checkpoint(config['EXP1_MODEL_1'], ckpt_dir)
        model2_params = load_checkpoint(config['EXP1_MODEL_2'], ckpt_dir)
    else: model1_params, model2_params = None, None



    rngs = jax.random.split(rng, config["NUM_SEEDS"])    
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(lambda rng: train_jit(rng, preloaded_params=loaded_ckpt,
                                         model1_params=model1_params, 
                                         model2_params=model2_params))(rngs)


    train_state = jax.tree_map(lambda x: x[0], out["runner_state"][0])
    
    # filename = f'{config["ENV_NAME"]}_{layout_name}'
    # state_seq, shaped_rewards_seq = get_rollout(train_state, config, model1_params=model1_params, model2_params=model2_params)
    # # viz = OvercookedVisualizer()
    # viz.animate(state_seq, shaped_rewards_seq, agent_view_size=5, filename=f"{filename}.mp4")
    # wandb.log({"animation": wandb.Video(f"{filename}.mp4", format="mp4")})


    if config['SAVE_MODEL']:
        model_name = config['SAVE_MODEL_NAME']
        save_checkpoint(train_state, model_name)

    print('fiinnnnnnnnnnnnnnnnnn')


if __name__ == "__main__":
    main()