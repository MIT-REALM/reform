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
from typing import Annotated, Literal

from reform.utils.typing import Obs, Action, PRNGKey, Params

from .module.encoder import encoder_modules
from .module.value import Value
from .module.actor import ActorVectorField
from .module.utils import ModuleDict, TrainState
from ..trainer.utils import has_any_nan_or_inf, compute_norm_and_clip


@Parameter(name="*", group="AgentConfig")
@dataclass
class FQLCfg:
    """Configuration for Flow Q-learning (FQL) agent.

    Parameters
    ----------
    lr : float
        Learning rate.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    tau : float
        Target network update rate.
    q_agg : Literal["min", "mean"]
        Method to aggregate Q-values from ensembles.
    gamma : float
        Discount factor.
    normalize_q_loss : bool
        Whether to normalize the Q loss.
    alpha : float
        W2 regularization weight (need to be tuned for each environment).
    encoder : str | None
        Name of the encoder module to use. If None, no encoder is used.
    actor_hidden_dims : list[int]
        Hidden dimensions for the actor network.
    value_hidden_dims : list[int]
        Hidden dimensions for the value network.
    actor_layer_norm : bool
        Whether to use layer normalization in the actor network.
    value_layer_norm : bool
        Whether to use layer normalization in the value network.
    action_flow_steps : int
        Number of flow steps for action sampling.
    action_dim : int | None
        Dimension of the action space (set during agent creation).
    obs_dims : tuple[int, ...] | None
        Dimensions of the observation space (set during agent creation).
    """
    # Training parameters.
    lr: float = 3e-4
    max_grad_norm: float = 10.0
    tau: float = 0.005
    q_agg: Literal["min", "mean"] = "mean"
    gamma: float = 0.995
    normalize_q_loss: bool = False
    alpha: float = 10.0

    # Network parameters.
    encoder: str | None = None
    actor_hidden_dims: Annotated[list[int], Parameter(consume_multiple=True)] = (512, 512, 512, 512)
    value_hidden_dims: Annotated[list[int], Parameter(consume_multiple=True)] = (512, 512, 512, 512)
    actor_layer_norm: bool = False
    value_layer_norm: bool = True

    # Action sampling parameters.
    action_flow_steps: int = 10
    action_dim: int | None = None
    obs_dims: tuple[int, ...] | None = None

    @property
    def agent_name(self):
        return "fql"


class FQLAgent(struct.PyTreeNode):
    """Flow Q-learning (FQL) agent."""
    network: TrainState
    cfg: FQLCfg = struct.field(pytree_node=False)

    @classmethod
    def create(
            cls,
            seed: int,
            ex_observations: Obs,
            ex_actions: Action,
            cfg: FQLCfg,
    ):
        key = jr.PRNGKey(seed)
        key, init_key = jr.split(key, 2)

        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]
        obs_dims = ex_observations.shape[1:]

        # Define encoders.
        encoders = dict()
        if cfg.encoder is not None:
            encoder_module = encoder_modules[cfg.encoder]
            encoders['critic'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=cfg.value_hidden_dims,
            layer_norm=cfg.value_layer_norm,
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=cfg.actor_hidden_dims,
            action_dim=action_dim,
            layer_norm=cfg.actor_layer_norm,
            encoder=encoders.get('actor_bc_flow'),
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=cfg.actor_hidden_dims,
            action_dim=action_dim,
            layer_norm=cfg.actor_layer_norm,
            encoder=encoders.get('actor_onestep_flow'),
        )
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions)),
        )

        # Add encoders to ModuleDict to make them separately callable.
        if encoders.get('actor_bc_flow') is not None:
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))

        # Initialize networks and optimizers.
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=cfg.lr)
        network_params = network_def.init(init_key, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        # Update cfg with environment information.
        cfg = dataclasses.replace(cfg, action_dim=action_dim, obs_dims=obs_dims)

        return cls(network=network, cfg=cfg)

    @jax.jit
    def update(self, batch: dict, key: PRNGKey):
        """Update the agent and return a new agent with information dictionary."""

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
        key, actor_key, critic_key = jr.split(key, 3)

        # Critic loss.
        critic_loss, critic_info = self.critic_loss(batch, params, critic_key)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        # Actor loss.
        actor_loss, actor_info = self.actor_loss(batch, params, actor_key)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def critic_loss(self, batch: dict, params: Params, key: PRNGKey):
        """Critic TD loss."""
        key, sample_key = jr.split(key)
        next_actions = self.sample_actions(batch['next_observations'], key=sample_key)
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        if self.cfg.q_agg == 'min':
            next_q = next_qs.min(axis=0)
        elif self.cfg.q_agg == 'mean':
            next_q = next_qs.mean(axis=0)
        else:
            raise ValueError(f"Unknown q_agg method: {self.cfg.q_agg}")

        target_q = batch['rewards'] + self.cfg.gamma * batch['masks'] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'q_ill': has_any_nan_or_inf(q),
            'target_q_ill': has_any_nan_or_inf(target_q),
        }

    def actor_loss(self, batch: dict, params: Params, key: PRNGKey):
        """FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape
        key, x_key, t_key = jr.split(key, 3)

        # BC flow loss.
        x_0 = jr.normal(x_key, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jr.uniform(t_key, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # Distillation loss.
        key, noise_key = jr.split(key)
        noises = jr.normal(noise_key, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
        actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, params=params)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # Q loss.
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.cfg.normalize_q_loss:
            lam = jax.lax.stop_gradient(1 / (jnp.abs(q).mean() + 1e-5))
            q_loss = lam * q_loss

        # Total loss.
        actor_loss = bc_flow_loss + self.cfg.alpha * distill_loss + q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'actor_loss_ill': has_any_nan_or_inf(actor_loss),
        }

    @jax.jit
    def compute_flow_actions(self, obs: Obs, noises: jnp.ndarray):
        """Compute actions from the BC flow model using the Euler method."""
        if self.cfg.encoder is not None:
            obs = self.network.select('actor_bc_flow_encoder')(obs)
        actions = noises

        def euler_step_(i_, actions_):
            t = jnp.full((*obs.shape[:-1], 1), i_ / self.cfg.action_flow_steps)
            vels = self.network.select('actor_bc_flow')(obs, actions_, t, is_encoded=True)
            actions_ = actions_ + vels / self.cfg.action_flow_steps
            return actions_

        actions = jax.lax.fori_loop(0, self.cfg.action_flow_steps, euler_step_, actions)
        actions = jnp.clip(actions, -1, 1)

        return actions

    @jax.jit
    def sample_actions(self, obs: Obs, key: PRNGKey):
        """Sample actions from the one-step policy."""
        action_key, noise_key = jax.random.split(key)
        noises = jr.normal(
            action_key,
            (
                *obs.shape[: -len(self.cfg.obs_dims)],
                self.cfg.action_dim,
            ),
        )
        actions = self.network.select('actor_onestep_flow')(obs, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions
