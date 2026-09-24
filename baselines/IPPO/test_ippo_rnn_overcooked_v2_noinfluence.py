""" 
Based on PureJaxRL Implementation of PPO
"""
import os
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DIR"] = "/scratch/gpfs/rm4057/wandb2"
os.environ["WANDB_CONFIG_DIR"] = "/scratch/gpfs/rm4057/wandb2"
os.environ["WANDB_CACHE_DIR"] = "/scratch/gpfs/rm4057/wandb2"
os.makedirs("/scratch/gpfs/rm4057/wandb2", exist_ok=True)
os.chdir("/scratch/gpfs/rm4057/wandb2")
import wandb
import h5py
import gc

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
from scipy.interpolate import UnivariateSpline
from itertools import product
import gzip


import hydra
from omegaconf import OmegaConf
from datetime import datetime
# import os
# import wandb
# os.environ["WANDB_DIR"] = "/home/rm4057/wandb2"
# os.environ["WANDB_DIR"] = os.path.expanduser("~/wandb2")
# os.environ["WANDB_DIR"] = "/scratch/gpfs/rm4057/wandb2"
import functools
import math
from typing import NamedTuple
from scipy.spatial.distance import jensenshannon

import matplotlib.pyplot as plt
from ippo_ff_overcooked_v2 import ActorCritic as ActorCritic_ff

import json
from jax.tree_util import tree_map
from flax.traverse_util import flatten_dict
from jax import lax


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



def get_rollout(train_state, config, network2_params=None, rnn_params=None, 
                model1_params=None, model2_params=None, AGENT_1_1_SPEED=1, AGENT_1_2_SPEED=1, 
                seed_num=0, layout_name=None):
    
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
    network2 = ActorCritic_ff(env.action_space().n, activation="tanh")
    network3 = ActorCritic_ff(env.action_space().n, activation="tanh")

    key = jax.random.PRNGKey(seed_num)
    key, key_r, key_a = jax.random.split(key, 3)

    init_x = (
        jnp.zeros(
            (1, 1, math.prod(env.observation_space().shape))
        ),
        jnp.zeros((1, 1)),
    )

    hstate_0 = ScannedRNN.initialize_carry(1, config["GRU_HIDDEN_DIM"])
    hstate_1 = ScannedRNN.initialize_carry(1, config["GRU_HIDDEN_DIM"])

    if rnn_params:
        network_params = rnn_params
    else:
        network_params = train_state.params

    obs, state = env.reset(key_r)
    last_done = jnp.zeros((1), dtype=bool) #<--
    model_status_1 = 0

    if network2_params:
         init_x = jnp.zeros(env.observation_space().shape)
         init_x = init_x.flatten()
         network2.init(key_a, init_x)   


    AGENT_0_COOLDOWN = 2
    carry_init = (key, hstate_0, hstate_1, state, obs, last_done, model_status_1, 0, 0)

    def step_fn(carry, t):
        key, h0, h1, st, obs, last_done_flag, mode, a0_cnt, a1_cnt = carry
        key, k0, k1, ks = jax.random.split(key, 4)
        obs0 = obs["agent_0"].flatten().reshape(1, *env.observation_space().shape)
        obs1 = obs["agent_1"].flatten().reshape(1, *env.observation_space().shape)
        h0, pi0, _ = network.apply(network_params, h0, (obs0[None, :], last_done_flag[None, :]))
        h1, pi1, _ = network.apply(network_params, h1, (obs1[None, :], last_done_flag[None, :]))
        a0_raw = pi0.sample(seed=k0)[0][0]
        action_0 = jnp.where(a0_cnt == 0, a0_raw, 4)
        a0_cnt_new = jnp.where(a0_cnt == 0, AGENT_0_COOLDOWN, a0_cnt - 1)
        switch = action_0 == 6
        mode = jnp.where(switch, 1 - mode, mode)
        mode = jnp.where(t >= 200, 1 - mode, mode)
        agent1_speed_target = jnp.where(mode == 0, AGENT_1_1_SPEED, AGENT_1_2_SPEED)
        a1_cnt = jnp.where(switch, agent1_speed_target, a1_cnt)
        obs_flat = obs1.flatten()
        pi_m1, _ = network3.apply(model1_params, obs_flat)
        pi_m2, _ = network3.apply(model2_params, obs_flat)
        a1_raw = jnp.where(mode == 0, pi_m1.sample(seed=k1), pi_m2.sample(seed=k1))
        action_1 = jnp.where(a1_cnt == 0, a1_raw, 4)
        a1_cnt_new = jnp.where(a1_cnt == 0, agent1_speed_target, a1_cnt - 1)
        obs, st, rw, dn, info = env.step(ks, st, {"agent_0": action_0, "agent_1": action_1})
        rw_val = rw['agent_0']
        partner_tuple = (AGENT_1_1_SPEED, AGENT_1_2_SPEED)
        return (key, h0, h1, st, obs, jnp.asarray([dn["__all__"]]), mode, a0_cnt_new, a1_cnt_new), (st, rw_val, mode, h0, partner_tuple)


    T = int(config["ENV_KWARGS"]["max_steps"])
    (state_seq, reward_seq, agent1_modes, hidden_state_0_seq, partner_tuple_seq) = lax.scan(step_fn, carry_init, jnp.arange(T))[1]


    def serialise_state(state):
        state_dict = state.__dict__ if hasattr(state, '__dict__') else dict(state)
        state_dict = jax.device_get(state_dict)
        return tree_map(lambda x: x.tolist() if hasattr(x, "tolist") else x, state_dict)  # Convert to native

    state_seq = [jax.tree_util.tree_map(lambda x: x[i], state_seq) for i in range(T)]
    reward_seq = [jax.tree_util.tree_map(lambda x: x[i], reward_seq) for i in range(T)]
    agent1_modes = [jax.tree_util.tree_map(lambda x: x[i], agent1_modes) for i in range(T)]
    hidden_state_0_seq = [jax.tree_util.tree_map(lambda x: x[i], hidden_state_0_seq) for i in range(T)]
    partner_tuple_seq = [jax.tree_util.tree_map(lambda x: x[i], partner_tuple_seq) for i in range(T)]

    serialised_state_seq = [serialise_state(s) for s in state_seq]

    hidden_data_state_0_seq = []
    for params in hidden_state_0_seq:
        if isinstance(params, dict):
            flat_params = flatten_dict(params, sep="/")
            param_data = {k: np.array(v).flatten().tolist() for k, v in flat_params.items()}
        else:
            param_data = np.array(params).flatten().tolist()
        hidden_data_state_0_seq.append(param_data)


    data_to_save = {
        "state_seq": serialised_state_seq,
        "rewards": [float(r) for r in reward_seq],
        "agent1_mode_seq": [float(a) for a in agent1_modes],
        "agent1_speed_seq": [[int(a), int(b)] for a, b in partner_tuple_seq],
        "hidden_state_0_seq": hidden_data_state_0_seq,
    }

    # env_name = config["ENV_NAME"]

    # artifact_name = f"noinfluence_rollout_data_seed_{seed_num}_env_name_{env_name}_speeds_{AGENT_1_1_SPEED}{AGENT_1_2_SPEED}.json"

    # with open(artifact_name, "w") as f:
    #     json.dump(data_to_save, f)

    # artifact = wandb.Artifact(artifact_name, type="rollout")
    # artifact.add_file(artifact_name)
    # wandb.log_artifact(artifact)


    # artifact_dir = "/scratch/gpfs/rm4057/rollouts"
    # os.makedirs(artifact_dir, exist_ok=True)
    # # artifact_basename = f"noinfluence_rollout_data_seed_{seed_num}_env_name_{env_name}_speeds_{AGENT_1_1_SPEED}{AGENT_1_2_SPEED}.json"
    artifact_basename = f"noinfluence_rollout_data_seed_{seed_num}_layout_name_{layout_name}_speeds_{AGENT_1_1_SPEED}{AGENT_1_2_SPEED}"
    # artifact_path = os.path.join(artifact_dir, artifact_basename)
    # # with open(artifact_path, "w") as f:
    # #     json.dump(data_to_save, f)
    # with gzip.open(artifact_path, "wt") as f:
    #     json.dump(data_to_save, f)    
    # artifact = wandb.Artifact(artifact_basename, type="rollout")
    # artifact.add_file(artifact_path, name=artifact_basename)
    # wandb.log_artifact(artifact)

    agent1_speeds = [f"Speeds: 1/{AGENT_1_1_SPEED+1} serve, 1/{AGENT_1_2_SPEED+1} onion"]*T
    return state_seq, reward_seq, agent1_modes, agent1_speeds, data_to_save, artifact_basename



def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}



@hydra.main(
    version_base=None, config_path="config", config_name="ippo_rnn_overcooked_v2"
)
def main(config):
    wandb_dir = "/scratch/gpfs/rm4057/wandb2"
    os.makedirs(wandb_dir, exist_ok=True)

    
    config = OmegaConf.to_container(config)

    layout_name = config["ENV_KWARGS"]["layout"]
    config["ENV_KWARGS"]["layout"] = overcooked_v2_layouts[layout_name]    
    num_seeds = config["NUM_SEEDS"]

    start_time = datetime.now()
    test_model = config["RNN_LOAD_MODEL_NAME"]

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN", "OvercookedV2"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"noinfluence_{test_model}_{layout_name}",
        dir=wandb_dir,
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


    ########################################
    # (0,6) -> (0,6)
    AGENT_1_1_SPEED, AGENT_1_2_SPEED = 0,6
    seed_num = 1

    speed_pairs_train = list(product([1,2,3],[4,7,8,9])) + list(product([4,7,8,9],[1,2,3]))
    speed_pairs_test = list(product([0,1,2,3],[5,6])) + list(product([7,8,9],[0])) + list(product([5,6],[0,1,2,3])) + list(product([0],[7,8,9]))
    speed_pairs = speed_pairs_train + speed_pairs_test

    all_rollouts = []

    seed_start, seed_end = 0, 20

    batch_filename = f"noinfluence_rollouts_batch_{layout_name}_seeds_{seed_start}_to_{seed_end}.h5"
    save_path = f"/scratch/gpfs/rm4057/rollouts/{batch_filename}_{os.getpid()}.h5"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        os.remove(save_path) 

    with h5py.File(save_path, "a", libver="latest") as f_h5:
        for seed_num in range(seed_start,seed_end):
            for AGENT_1_1_SPEED, AGENT_1_2_SPEED in speed_pairs:
                    
                state_seq, reward_seq, agent1_modes, agent1_speeds, rollout_data, rollout_key = get_rollout(train_state, config, loaded_ckpt, loaded_ckpt_RNN, model1_params=model1_params, model2_params=model2_params,
                                                    AGENT_1_1_SPEED=AGENT_1_1_SPEED, AGENT_1_2_SPEED=AGENT_1_2_SPEED, seed_num=seed_num, layout_name=layout_name)

                if rollout_key in f_h5:
                    del f_h5[rollout_key]

                g = f_h5.create_group(rollout_key)

                g.create_dataset(
                    "rewards",
                    data=np.asarray(rollout_data["rewards"], dtype=np.float32),
                    compression="gzip",
                    chunks=True,
                )
                g.create_dataset(
                    "agent1_mode_seq",
                    data=np.asarray(rollout_data["agent1_mode_seq"], dtype=np.float32),
                    compression="gzip",
                    chunks=True,
                )
                g.create_dataset(
                    "agent1_speed_seq",
                    data=np.asarray(rollout_data["agent1_speed_seq"], dtype=np.int32),
                    compression="gzip",
                    chunks=True,
                )

                flat_hidden = [
                    (np.asarray(list(h.values())) if isinstance(h, dict) else np.asarray(h)).ravel()
                    for h in rollout_data["hidden_state_0_seq"]
                ]
                g.create_dataset(
                    "hidden_state_0_seq",
                    data=np.asarray(flat_hidden, dtype=np.float32),
                    compression="gzip",
                    chunks=True,
                )

                f_h5.flush() 
                del rollout_data  
                gc.collect()


    clock2 = datetime.now() - start_time
    wandb.log({"clocktime2": clock2.total_seconds()})

    artifact = wandb.Artifact(batch_filename, type="rollout_batch")
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)


    # viz = OvercookedVisualizer()
    # viz.animate(state_seq, reward_seq, agent_view_size=5, filename=f"{filename}.mp4", agent1_modes=agent1_modes, agent1_speeds=agent1_speeds)
    # wandb.log({f"animation_{AGENT_1_1_SPEED}{AGENT_1_2_SPEED}_{AGENT_2_1_1_SPEED}{AGENT_2_1_2_SPEED}": wandb.Video(f"{filename}.mp4", format="mp4")})




if __name__ == "__main__":
    main()