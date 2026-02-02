import flax.linen as nn
import jax.numpy as jnp
import distrax

from typing import Sequence, Optional

from reform.utils.typing import Obs, Action, FloatScalar
from reform.utils.networks import MLP, default_init
from .distribution import TanhMultivariateNormalDiag


class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations: Obs, actions: Action, times: FloatScalar = None, is_encoded: bool = False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)(inputs)

        return v


class StdTanhNormalPolicy(nn.Module):
    """Tanh normal policy.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
        log_std_min: Lower bound of standard deviation.
        log_std_max: Upper bound of standard deviation.
        low: Lower bound of action space.
        high: Upper bound of action space.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    low: Optional[float] = None
    high: Optional[float] = None

    @nn.compact
    def __call__(self, observations: Obs) -> distrax.Distribution:
        """
        Return the action distribution given observations.

        Args:
            observations: Observations.
        """
        if self.encoder is not None:
            observations = self.encoder(observations)

        outputs = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)(observations)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        log_stds = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = TanhMultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds), low=self.low, high=self.high)
        return distribution
