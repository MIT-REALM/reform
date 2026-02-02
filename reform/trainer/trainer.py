import dataclasses
import yaml
import wandb
import os
import datetime
import gymnasium as gym
import numpy as np
import jax.random as jr

from dataclasses import dataclass
from cyclopts import Parameter
from tqdm import tqdm

from reform.trainer.datasets import Dataset, DatasetCfg
from reform.agents import Agent
from reform.trainer.utils import supply_rng, test_actor, save_agent
from reform.utils.typing import PRNGKey


@Parameter(name="*", group="TrainerConfig")
@dataclass
class TrainerCfg:
    """Configuration for trainer.
    
    Parameters
    ----------
    steps : int
        Total number of training steps.
    eval_interval : int
        Interval (in steps) for evaluation on the validation dataset.
    test_interval : int
        Interval (in steps) for testing in the environment.
    test_epi : int
        Number of episodes for testing.
    save_interval : int
        Interval (in steps) for saving the model.
    batch_size : int
        Batch size for training.
    save_dir : str
        Directory to save logs and models.
    run_name : str | None
        Optional name for the run.
    """
    steps: int = 5000000
    eval_interval: int = 10000
    test_interval: int = 50000
    test_epi: int = 20
    save_interval: int = 100000
    batch_size: int = 256
    save_dir: str = "logs"
    run_name: str | None = None


class Trainer:

    def __init__(
            self,
            cfg: TrainerCfg,
            env_name: str,
            env: gym.Env,
            train_dataset: Dataset,
            val_dataset: Dataset,
            dataset_cfg: DatasetCfg,
            agent: Agent,
    ):
        self.cfg = cfg
        self.env_name = env_name
        self.env = env
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.dataset_cfg = dataset_cfg
        self.agent = agent

    def train(self, seed: int, debug: bool = False):
        # Record start time.
        start_time = datetime.datetime.now()

        # Create random key.
        key = jr.PRNGKey(seed)

        # Create save directory.
        log_dir = os.path.join(self.cfg.save_dir, self.env_name, self.agent.cfg.agent_name)
        if self.cfg.run_name is not None:
            log_dir = os.path.join(log_dir, f"seed{seed}_{start_time.strftime('%m%d%H%M%S')}_{self.cfg.run_name}")
        else:
            if hasattr(self.agent.cfg, "alpha"):
                log_dir = os.path.join(log_dir, f"seed{seed}_{start_time.strftime('%m%d%H%M%S')}_alpha{self.agent.cfg.alpha}")
                self.cfg.run_name = f"alpha{self.agent.cfg.alpha}"
            else:
                log_dir = os.path.join(log_dir, f"seed{seed}_{start_time.strftime('%m%d%H%M%S')}")
        model_dir = os.path.join(log_dir, "models")

        if not debug:
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)

            # Set up wandb.
            config = dataclasses.asdict(self.cfg)
            config.update({
                'env_name': self.env_name,
                'agent_name': self.agent.cfg.agent_name,
            })
            config.update(dataclasses.asdict(self.agent.cfg))
            config.update(dataclasses.asdict(self.dataset_cfg))
            run_name = f"seed{seed}_{self.agent.cfg.agent_name}"
            run_name += f"_{self.cfg.run_name}" if self.cfg.run_name else ""
            wandb.init(
                project="reform",
                dir=log_dir,
                group=self.env_name,
                name=run_name,
                config=config,
            )
            yaml.dump(config, open(os.path.join(log_dir, "config.yaml"), "w"))

        # Start training.
        pbar = tqdm(total=self.cfg.steps, desc="Training", ncols=80)
        steps_len = len(str(self.cfg.steps))
        for step in range(self.cfg.steps + 1):
            # Test the agent in the environment.
            if step % self.cfg.test_interval == 0:
                pbar.set_description(f"Testing")
                key, test_key = jr.split(key, 2)
                test_info = self.test(test_key)
                if not debug:
                    wandb.log(test_info, step=step)
                tqdm.write(f"Step: {step:>{steps_len}}, " + ", ".join([f"{k}: {v:8.2f}" for k, v in test_info.items()]))

            # Evaluate on the validation dataset.
            if step % self.cfg.eval_interval == 0:
                pbar.set_description(f"Evaluating")
                key, eval_key = jr.split(key, 2)
                val_batch = self.val_dataset.sample(self.cfg.batch_size)
                _, val_info = self.agent.total_loss(batch=val_batch, params=None, key=eval_key)
                val_info = {f"eval/{k}": v for k, v in val_info.items()}
                if not debug:
                    wandb.log(val_info, step=step)

            # Save model.
            if step % self.cfg.save_interval == 0 and not debug:
                pbar.set_description(f"Saving model")
                save_agent(self.agent, model_dir, step)

            # Sample a batch from the training dataset.
            pbar.set_description(f"Training")
            batch = self.train_dataset.sample(self.cfg.batch_size)

            # Perform a training step.
            key, update_key = jr.split(key)
            self.agent, update_info = self.agent.update(batch, update_key)
            update_info = {f"train/{k}": float(v) for k, v in update_info.items()}
            if not debug and step % self.cfg.eval_interval == 0:
                wandb.log(update_info, step=step)

            # Update progress bar.
            pbar.update(1)

    def test(self, key: PRNGKey):
        """Test the agent in the environment."""
        actor_fn = supply_rng(self.agent.sample_actions, key=key)

        trajs, _ = test_actor(
            actor_fn,
            self.env,
            n_episodes=self.cfg.test_epi,
            rng=np.random.default_rng(int(sum(key))),
            render=False,
            verbose=False
        )

        traj_rewards = [np.sum(traj['reward']) for traj in trajs]
        traj_success = [traj['info'][-1]['success'] for traj in trajs]

        return {
            'test/reward_mean': np.mean(traj_rewards),
            'test/reward_max': np.max(traj_rewards),
            'test/reward_min': np.min(traj_rewards),
            'test/success_rate': np.mean(traj_success),
        }
