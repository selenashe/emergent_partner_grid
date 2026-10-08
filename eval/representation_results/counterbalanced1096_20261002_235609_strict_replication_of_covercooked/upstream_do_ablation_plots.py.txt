from collections import Counter, defaultdict
import functools
import json
import os
import gzip
import shutil

import chex
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from utils import (
    save_fig,
    load_rollout,
    embed_hidden_states,
    relabel_speeds,
    dotdict,
    load_rollouts_batch,
)

colors = ["xkcd:green blue", "xkcd:periwinkle", "xkcd:pumpkin"]


class LinearProbe(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.num_classes)(x)
        return x


@functools.partial(jax.jit, static_argnums=(3,))
def train_step(ts, embeddings, labels, n_cats):
    def loss_fn(params):
        logits = ts.apply_fn(params, embeddings)
        losses = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels
        )
        return losses.mean()

    loss, grads = jax.value_and_grad(loss_fn)(ts.params)
    return ts.apply_gradients(grads=grads), loss


@jax.jit
def do_eval(ts, embeddings, labels):
    logits = ts.apply_fn(ts.params, embeddings)
    predictions = jnp.argmax(logits, axis=1)

    mae = jnp.abs(predictions - labels)
    e_max = labels.max() - labels.min()
    accs = 1 - (mae / e_max)
    return accs.mean()


def shuffle(rng, embeddings, labels):
    perm = jax.random.permutation(rng, embeddings.shape[0])
    return embeddings[perm], labels[perm]


def split(embeddings, labels, frac=0.8):
    train_idxs, test_idxs, label_counts = [], [], Counter(labels)
    for label, count in label_counts.items():
        idxs = np.where(labels == label)[0]
        idxs = np.random.permutation(idxs)
        train_idxs.extend(idxs[: int(count * frac)])
        test_idxs.extend(idxs[int(count * frac) :])

    train_idxs, test_idxs = np.array(train_idxs), np.array(test_idxs)
    return (
        embeddings[train_idxs],
        labels[train_idxs],
        embeddings[test_idxs],
        labels[test_idxs],
    )


# @functools.partial(jax.jit, static_argnums=(3, 4, 5))
def train_probe(
    rng,
    x,
    y,
    n_iterations=int(1e3),
    lr=1e-2,
    train_frac=0.8,
    return_best=True,
    params=None,
):
    x, y = shuffle(rng, x, y)
    x_train, y_train, x_test, y_test = split(x, y, train_frac)

    n_classes = len(jnp.unique(y.reshape(-1)))
    model = LinearProbe(num_classes=n_classes)
    if params is None:
        params = model.init(rng, jnp.empty([1, x.shape[1]]))
    tx = optax.adamw(learning_rate=lr, weight_decay=1e-3)
    ts = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    best_acc, best_ts = 0.0, ts

    for itr in range(n_iterations + 1):
        rng, _ = jax.random.split(rng)
        x_train, y_train = shuffle(rng, x_train, y_train)
        ts, loss = train_step(ts, x_train, y_train, n_classes)
        if itr % 20 == 0:
            acc = do_eval(ts, x_test, y_test)
            if acc > best_acc:
                best_acc = acc
                best_ts = ts

    if return_best:
        return best_ts, loss, best_acc
    else:
        acc = do_eval(ts, x_test, y_test)
        return ts, loss, acc


def get_probs(ts_1, ts_2, embeddings, labels, label):
    idxs = jnp.where(jnp.all(labels == label, axis=1))
    x = embeddings[idxs]
    logits_1 = ts_1.apply_fn(ts_1.params, x)
    logits_2 = ts_2.apply_fn(ts_2.params, x)
    probs_1, probs_2 = jax.nn.softmax(logits_1), jax.nn.softmax(logits_2)
    return jnp.einsum("i,j->ij", probs_1.mean(0), probs_2.mean(0))


rollout_dicts = defaultdict(dict)
artifact_names = [x for x in os.listdir("artifacts") if "rollouts" in x]
for an in tqdm(artifact_names, desc="Loading rollouts"):
    if "ff" in an:
        continue
    batch_type = an.split("_")[0]
    batch_type = "basic" if batch_type == "rollouts" else batch_type

    layout = an.split("batch_")[1].split("_seeds")[0]
    rollouts = load_rollouts_batch(an)
    rollout_dicts[batch_type][layout] = rollouts

rng = jax.random.PRNGKey(0)
method = "UMAP"
method_kwargs = {"min_dist": 1.0, "n_neighbors": 919}

# DO EMBEDDING PLOTS
sns.set_style("dark")
fig, axs = plt.subplots(3, 5, figsize=(15, 9))
cmap = sns.color_palette("Spectral", as_cmap=True)

conditions = ["basic", "single", "noinfluence"]
layouts = sorted(list(rollout_dicts[conditions[0]].keys()))

for i, condition in enumerate(conditions):
    for j, layout in enumerate(layouts):
        try:
            rollouts = rollout_dicts[condition][layout]
            hs = np.concatenate(
                [
                    r.hidden_state_0_seq[-50:].mean(axis=0).reshape(1, -1)
                    for r in rollouts
                ]
            )
            labels = np.concatenate(
                [r.agent1_speed_seq[-1].reshape(1, -1) for r in rollouts]
            )
            labels = labels[:, 0] - labels[:, 1]
            embeddings, _ = embed_hidden_states(hs, method, method_kwargs)
            sns.scatterplot(
                x=embeddings[:, 0],
                y=embeddings[:, 1],
                hue=labels,
                ax=axs[i, j],
                legend=False,
                palette=cmap,
                s=50,
                linewidth=0.0,
                alpha=0.85,
            )
        except Exception:
            pass
        axs[i, j].set(
            xlabel="",
            xticks=[],
            xticklabels=[],
            ylabel=condition if j == 0 else "",
            yticks=[],
            yticklabels=[],
            title=layout if i == 0 else "",
        )
        sns.despine(ax=axs[i, j], bottom=True, left=True)

save_fig(fig, "ablation/embeddings")

# DO LILNEAR PROBE ANALYSIS
data = []
for i, condition in enumerate(conditions):
    for layout in layouts:
        rollouts = rollout_dicts[condition][layout]
        labels = np.concatenate(
            [r.agent1_speed_seq[-1].reshape(1, -1) for r in rollouts]
        )
        params = [None, None, None]
        for t in tqdm(
            [1, 50, 100, 150, 200, 250, 300, 350, 400], desc=f"{condition}:{layout}"
        ):
            hs = np.concatenate(
                [r.hidden_state_0_seq[:t].mean(axis=0).reshape(1, -1) for r in rollouts]
            )
            ts_best, _, acc_best = train_probe(
                rng,
                hs,
                labels.argmax(axis=1),
                params=params[0],
            )
            ts_1, _, acc_1 = train_probe(rng, hs, labels[:, 0], params=params[1])
            ts_2, _, acc_2 = train_probe(rng, hs, labels[:, 1], params=params[2])
            params = [ts_best.params, ts_1.params, ts_2.params]
            data.append(
                {
                    "condition": condition,
                    "layout": layout,
                    "t": t,
                    "acc_best": float(acc_best),
                    "acc_1": float(acc_1),
                    "acc_2": float(acc_2),
                    "acc_avg": float((acc_1 + acc_2) / 2),
                }
            )
data = pd.DataFrame(data)
data.to_csv("analysis/ablation_probe.csv")

# random baseline
x = np.array(jax.random.normal(rng, (920, 128)))
y = np.array(jax.random.randint(rng, (920,), 0, 10))
_, _, random_acc = train_probe(rng, x, y)

sns.set_style("white")
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
sns.lineplot(
    data=data,
    x="t",
    y="acc_1",
    hue="condition",
    palette=colors,
    ax=axs[0],
    legend=False,
    linewidth=4,
)
axs[0].set(
    ylim=(0.60, 0.95),
    ylabel="Test accuracy (distance-aware)",
    xlabel="Timestep",
    title="Partner's task 1 speed",
)
sns.despine(ax=axs[0])
axs[0].axhline(random_acc, color="red", linestyle="--", linewidth=3)
sns.lineplot(
    data=data,
    x="t",
    y="acc_2",
    hue="condition",
    palette=colors,
    ax=axs[1],
    legend=False,
    linewidth=4,
)
axs[1].set(
    ylim=(0.60, 0.95),
    ylabel="",
    yticks=[],
    yticklabels=[],
    xlabel="Timestep",
    title="Partner's task 2 speed",
)
axs[1].axhline(random_acc, color="red", linestyle="--", linewidth=3)
sns.despine(ax=axs[1], left=True)
save_fig(fig, "ablation/probe")
