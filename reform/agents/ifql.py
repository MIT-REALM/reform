import dataclasses
import jax.random as jr
import jax
import copy
import optax
import flax.linen as nn
import jax.numpy as jnp

from dataclasses import dataclass
from flax import struct
from cyclopts import Parameter
from typing import Annotated

from reform.utils.typing import Obs, Action, PRNGKey, Params

from .module.encoder import encoder_modules
from .module.value import Value
from .module.actor import ActorVectorField
from .module.utils import ModuleDict, TrainState
from .utils import expectile_loss
from ..trainer.utils import has_any_nan_or_inf, compute_norm_and_clip


@Parameter(name="*", group="AgentConfig")
@dataclass
class IFQLCfg:
    """Configuration for implicit flow Q-learning (IFQL) agent.

    Parameters
    ----------
    lr : float
        Learning rate.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    tau : float
        Target network update rate.
    expectile : float
        Expectile for the value loss.
    gamma : float
        Discount factor.
    encoder : str | None
        Name of the encoder to use. If None, no encoder is used.
    actor_hidden_dims : list[int]
        Hidden dimensions for the actor network.
    value_hidden_dims : list[int]
        Hidden dimensions for the value and critic networks.
    actor_layer_norm : bool
        Whether to use layer normalization in the actor network.
    value_layer_norm : bool
        Whether to use layer normalization in the value and critic networks.
    num_action_samples : int
        Number of action samples for action selection.
    action_flow_steps : int
        Number of flow steps for action sampling.
    action_dim : int | None
        Dimension of the action space (set during agent creation).
    """
    # Training parameters.
    lr: float = 3e-4
    max_grad_norm: float = 10.0
    tau: float = 0.005
    expectile: float = 0.9
    gamma: float = 0.995

    # Network parameters.
    encoder: str | None = None
    actor_hidden_dims: Annotated[list[int], Parameter(consume_multiple=True)] = (512, 512, 512, 512)
    value_hidden_dims: Annotated[list[int], Parameter(consume_multiple=True)] = (512, 512, 512, 512)
    actor_layer_norm: bool = False
    value_layer_norm: bool = True

    # Action sampling parameters.
    num_action_samples: int = 32
    action_flow_steps: int = 10
    action_dim: int | None = None

    @property
    def agent_name(self):
        return "ifql"


class IFQLAgent(struct.PyTreeNode):
    """Implicit flow Q-learning (IFQL) agent.

    IFQL is the flow variant of implicit diffusion Q-learning (IDQL).
    """
    network: TrainState
    cfg: IFQLCfg = struct.field(pytree_node=False)

    @classmethod
    def create(
            cls,
            seed: int,
            ex_observations: Obs,
            ex_actions: Action,
            cfg: IFQLCfg,
    ):
        key = jr.PRNGKey(seed)
        key, init_key = jr.split(key, 2)

        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if cfg.encoder is not None:
            encoder_module = encoder_modules[cfg.encoder]
            encoders['value'] = encoder_module()
            encoders['critic'] = encoder_module()
            encoders['actor_flow'] = encoder_module()

        # Define networks.
        value_def = Value(
            hidden_dims=cfg.value_hidden_dims,
            layer_norm=cfg.value_layer_norm,
            num_ensembles=1,
            encoder=encoders.get('value'),
        )
        critic_def = Value(
            hidden_dims=cfg.value_hidden_dims,
            layer_norm=cfg.value_layer_norm,
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_flow_def = ActorVectorField(
            hidden_dims=cfg.actor_hidden_dims,
            action_dim=action_dim,
            layer_norm=cfg.actor_layer_norm,
            encoder=encoders.get('actor_flow'),
        )
        network_info = dict(
            value=(value_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_flow=(actor_flow_def, (ex_observations, ex_actions, ex_times)),
        )

        # Add encoders to ModuleDict to make them separately callable.
        if encoders.get('actor_flow') is not None:
            network_info['actor_flow_encoder'] = (encoders.get('actor_flow'), (ex_observations,))

        # Initialize networks and optimizers.
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=cfg.lr)
        network_params = network_def.init(init_key, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # Update cfg with environment information.
        cfg = dataclasses.replace(cfg, action_dim=action_dim)

        return cls(network=network, cfg=cfg)

    @jax.jit
    def update(self, batch: dict, key: PRNGKey):

        def loss_fn_(params):
            total_loss_, info_ = self.total_loss(batch, params, key)
            return total_loss_, info_

        grad, info = jax.grad(loss_fn_, has_aux=True)(self.network.params)
        grad_ill = has_any_nan_or_inf(grad)
        grad, grad_norm = compute_norm_and_clip(grad, self.cfg.max_grad_norm)
        new_network = self.network.apply_gradients(grads=grad)

        # Update target network.
        self.target_update(new_network, "critic")

        return self.replace(network=new_network), info | {"total/grad_norm": grad_norm, "total/grad_ill": grad_ill}

    def target_update(self, network: nn.Module, module_name: str):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.cfg.tau + tp * (1 - self.cfg.tau),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    def total_loss(self, batch: dict, params: Params, key: PRNGKey):
        info = {}

        # Value loss.
        value_loss, value_info = self.value_loss(batch, params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        # Critic loss.
        critic_loss, critic_info = self.critic_loss(batch, params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        # Actor loss.
        key, actor_key = jax.random.split(key)
        actor_loss, actor_info = self.actor_loss(batch, params, actor_key)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + critic_loss + actor_loss
        return loss, info

    def value_loss(self, batch: dict, params: Params):
        """IQL value loss."""
        q1, q2 = self.network.select('target_critic')(batch['observations'], actions=batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(batch['observations'], params=params)
        value_loss = expectile_loss(q - v, q - v, self.cfg.expectile).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'v_ill': has_any_nan_or_inf(v),
        }

    def critic_loss(self, batch: dict, params: Params):
        """IQL critic loss."""
        next_v = self.network.select('value')(batch['next_observations'])
        q = batch['rewards'] + self.cfg.gamma * batch['masks'] * next_v

        q1, q2 = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=params)
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'q_ill': has_any_nan_or_inf(q),
        }

    def actor_loss(self, batch: dict, params: Params, key: PRNGKey):
        """BC flow-matching actor loss."""
        batch_size, action_dim = batch['actions'].shape
        key, x_key, t_key = jax.random.split(key, 3)

        x_0 = jr.normal(x_key, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jr.uniform(t_key, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.network.select('actor_flow')(batch['observations'], x_t, t, params=params)
        actor_loss = jnp.mean((pred - vel) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'actor_loss_ill': has_any_nan_or_inf(actor_loss),
        }

    @jax.jit
    def sample_actions(self, obs: Obs, key: PRNGKey):
        """Sample actions from the actor."""
        orig_obs = obs
        if self.cfg.encoder is not None:
            obs = self.network.select('actor_flow_encoder')(obs)
        action_key, noise_key = jr.split(key)

        # Sample `num_samples` noises and propagate them through the flow.
        actions = jr.normal(
            action_key,
            (
                *obs.shape[:-1],
                self.cfg.num_action_samples,
                self.cfg.action_dim,
            ),
        )
        n_obs = jnp.repeat(jnp.expand_dims(obs, 0), self.cfg.num_action_samples, axis=0)
        n_orig_obs = jnp.repeat(jnp.expand_dims(orig_obs, 0), self.cfg.num_action_samples, axis=0)

        def euler_step_(i_, actions_):
            t = jnp.full((*obs.shape[:-1], self.cfg.num_action_samples, 1), i_ / self.cfg.action_flow_steps)
            vels = self.network.select('actor_flow')(n_obs, actions_, t, is_encoded=True)
            actions_ = actions_ + vels / self.cfg.action_flow_steps
            return actions_

        actions = jax.lax.fori_loop(0, self.cfg.action_flow_steps, euler_step_, actions)
        actions = jnp.clip(actions, -1, 1)

        # Pick the action with the highest Q-value.
        q = self.network.select('critic')(n_orig_obs, actions=actions).min(axis=0)
        actions = actions[jnp.argmax(q)]
        return actions
