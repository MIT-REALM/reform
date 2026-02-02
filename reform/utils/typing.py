from flax.core import FrozenDict
from jaxtyping import Array, Bool, Float, Int, Shaped
from typing import Dict, Any


# Jax types.
PRNGKey = Float[Array, '2']
BoolScalar = Bool[Array, ""]
Shape = tuple[int, ...]
BFloat = Float[Array, "b"]
BInt = Int[Array, "b"]
FloatScalar = float | Float[Array, ""]
IntScalar = int | Int[Array, ""]
TFloat = Float[Array, "T"]

# Environment types.
Action = Float[Array, 'action_dim']
Reward = FloatScalar
Done = BoolScalar
Info = Dict[str, Shaped[Array, '']]
Obs = Float[Array, 'obs_dim']
State = Float[Array, 'state_dim']

# Neural network types.
Params = dict[str, Any] | FrozenDict[str, Any]

