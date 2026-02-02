import jax.tree_util as jtu
import numpy as np
import functools as ft
import jax
import jax.numpy as jnp

from flax.core.frozen_dict import FrozenDict
from cyclopts import Parameter
from dataclasses import dataclass


def get_size(data):
    """Return the size of the dataset."""
    sizes = jtu.tree_map(lambda arr: len(arr), data)
    return max(jtu.tree_leaves(sizes))


@ft.partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@ft.partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


@Parameter(name="*", group="DatasetConfig")
@dataclass
class DatasetCfg:
    """Configuration for dataset.

    Parameters
    ----------
    p_aug : float | None
        Probability of applying image augmentation.
    frame_stack : int | None
        Number of frames to stack.
    """
    p_aug: float | None = None
    frame_stack: int | None = None


class Dataset(FrozenDict):

    @classmethod
    def create(cls, cfg: DatasetCfg, seed: int, freeze=True, **fields):
        data = fields
        assert 'observations' in data
        if freeze:
            jtu.tree_map(lambda arr: arr.setflags(write=False), data)
        rng = np.random.default_rng(seed)
        return cls(cfg, rng, data)

    def __init__(self, cfg: DatasetCfg, rng: np.random.Generator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.cfg = cfg
        self.rng = rng
        # self.return_next_actions = False  # Whether to additionally return next actions; set outside the class.

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_indices(self, num_indices: int) -> np.ndarray:
        """Return `num_indices` random indices."""
        # return np.random.randint(self.size, size=num_indices)
        return self.rng.integers(0, self.size, size=num_indices)

    def sample(self, batch_size: int, indices=None) -> dict:
        """Sample a batch of transitions."""
        if indices is None:
            indices = self.get_random_indices(batch_size)
        batch = self.get_subset(indices)
        if self.cfg.frame_stack is not None:
            # Stack frames.
            initial_state_indices = self.initial_locs[np.searchsorted(self.initial_locs, indices, side='right') - 1]
            obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
            next_obs = []  # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]].
            for i in reversed(range(self.cfg.frame_stack)):
                # Use the initial state if the index is out of bounds.
                cur_indices = np.maximum(indices - i, initial_state_indices)
                obs.append(jtu.tree_map(lambda arr: arr[cur_indices], self['observations']))
                if i != self.cfg.frame_stack - 1:
                    next_obs.append(jtu.tree_map(lambda arr: arr[cur_indices], self['observations']))
            next_obs.append(jtu.tree_map(lambda arr: arr[indices], self['next_observations']))

            batch['observations'] = jtu.tree_map(lambda *args: np.concatenate(args, axis=-1), *obs)
            batch['next_observations'] = jtu.tree_map(lambda *args: np.concatenate(args, axis=-1), *next_obs)
        if self.cfg.p_aug is not None:
            # Apply random-crop image augmentation.
            if self.rng.random() < self.cfg.p_aug:
                self.augment(batch, ['observations', 'next_observations'])
        return batch

    def get_subset(self, indices) -> dict:
        """Return a subset of the dataset given the indices."""
        result = jtu.tree_map(lambda arr: arr[indices], self._dict)
        # if self.return_next_actions:
        #     # WARNING: This is incorrect at the end of the trajectory. Use with caution.
        #     result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = self.rng.integers(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jtu.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )
