import socket
import jax.numpy as jnp
import jax.tree_util as jtu
import gymnasium as gym
import numpy as np
import jax.random as jr
import flax
import os
import pickle
import glob

from collections import defaultdict
from typing import Callable
from tqdm import tqdm


def internet(host="8.8.8.8", port=53, timeout=3):
    """
    Host: 8.8.8.8 (google-public-dns-a.google.com)
    OpenPort: 53/tcp
    Service: domain (DNS/TCP)
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        print(ex)
        return False


def is_connected():
    return internet()


def has_nan(x):
    return jtu.tree_map(lambda y: jnp.isnan(y).any(), x)


def has_any_nan(x):
    return jnp.array(jtu.tree_flatten(has_nan(x))[0]).any()


def has_inf(x):
    return jtu.tree_map(lambda y: jnp.isinf(y).any(), x)


def has_any_inf(x):
    return jnp.array(jtu.tree_flatten(has_inf(x))[0]).any()


def has_any_nan_or_inf(x):
    return has_any_nan(x) | has_any_inf(x)


def compute_norm(grad):
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jtu.tree_leaves(grad)))


def compute_norm_and_clip(grad, max_norm: float):
    g_norm = compute_norm(grad)
    clipped_g_norm = jnp.maximum(max_norm, g_norm)
    clipped_grad = jtu.tree_map(lambda t: (t / clipped_g_norm) * max_norm, grad)

    return clipped_grad, g_norm


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def test_actor(
        actor_fn: Callable,
        env: gym.Env,
        n_episodes: int,
        rng: np.random.Generator,
        video_frame_skip: int = 3,
        render: bool = False,
        verbose: bool = False
):
    trajs = []
    frames = []

    if verbose:
        pbar = tqdm(total=n_episodes, desc="Testing", ncols=80)
    else:
        pbar = None

    print_len = len(str(n_episodes))
    for i_traj in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(2**31)))
        done = False

        traj = defaultdict(list)
        frames_traj = []
        step = 0

        while not done:
            action = actor_fn(obs)
            action = np.clip(action, -1, 1)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            transition = dict(
                observation=obs,
                next_observation=next_obs,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)

            if render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                frames_traj.append(frame)

            obs = next_obs

        trajs.append(traj)
        frames.append(frames_traj)
        if verbose:
            tqdm.write(f"Episode {i_traj:>{print_len}}: "
                       f"Success: {str(bool(traj['info'][-1]['success'])).rjust(5)}, "
                       f"Reward: {jnp.sum(jnp.array(traj['reward'])):8.2f}, "
                       f"Length: {len(traj['reward']):>4}")
            pbar.update(1)

    return trajs, frames


def supply_rng(f, key=jr.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal key
        key, use_key = jr.split(key)
        return f(*args, key=use_key, **kwargs)

    return wrapped


def save_agent(agent, save_dir, epoch):
    """Save the agent to a file.

    Args:
        agent: Agent.
        save_dir: Directory to save the agent.
        epoch: Epoch number.
    """

    save_dict = dict(
        agent=flax.serialization.to_state_dict(agent),
    )
    save_path = os.path.join(save_dir, f'params_{epoch}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)


def restore_agent(agent, restore_path, restore_epoch):
    """Restore the agent from a file.

    Args:
        agent: Agent.
        restore_path: Path to the directory containing the saved agent.
        restore_epoch: Epoch number.
    """
    candidates = glob.glob(restore_path)

    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

    restore_path = candidates[0] + f'/params_{restore_epoch}.pkl'

    with open(restore_path, 'rb') as f:
        load_dict = pickle.load(f)

    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])

    print(f'Restored from {restore_path}')

    return agent
