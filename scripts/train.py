import cyclopts
import ipdb
import os
import numpy as np

from reform.trainer.utils import is_connected
from reform.agents import agents, AgentCfg, FQLCfg, IFQLCfg, DSRLCfg, ReFORMCfg
from reform.trainer.datasets import Dataset, DatasetCfg
from reform.trainer.trainer import TrainerCfg, Trainer
from reform.env import make_env_and_datasets


app = cyclopts.App()


def _train(
        agent_cfg: AgentCfg,
        trainer_cfg: TrainerCfg,
        dataset_cfg: DatasetCfg = DatasetCfg(),
        env_name: str = "cube-single-noisy-singletask-task1-v0",
        seed: int = 0,
        debug: bool = False
):
    # Set up environment variables and seed.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if not is_connected():
        os.environ["WANDB_MODE"] = "offline"
    if debug:
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["JAX_DISABLE_JIT"] = "True"
    global_rng = np.random.default_rng(seed)
    np.random.seed(global_rng.integers(2 ** 31))

    # Create environment and load datasets.
    env, train_dataset, val_dataset = make_env_and_datasets(
        env_name, frame_stack=dataset_cfg.frame_stack, render_mode="rgb_array"
    )
    train_dataset = Dataset.create(dataset_cfg, global_rng.integers(2**31), **train_dataset)
    val_dataset = Dataset.create(dataset_cfg, global_rng.integers(2**31), **val_dataset)

    # Create agent.
    example_batch = train_dataset.sample(1)
    agent_class = agents[agent_cfg.agent_name]
    agent = agent_class.create(
        global_rng.integers(2**31),
        example_batch['observations'],
        example_batch['actions'],
        agent_cfg,
    )

    # Initialize trainer.
    trainer = Trainer(trainer_cfg, env_name, env, train_dataset, val_dataset, dataset_cfg, agent)

    # Start training.
    trainer.train(seed, debug)


@app.command
def fql(
        agent_cfg: FQLCfg = FQLCfg(),
        trainer_cfg: TrainerCfg = TrainerCfg(),
        dataset_cfg: DatasetCfg = DatasetCfg(),
        env_name: str = "cube-single-noisy-singletask-task1-v0",
        seed: int = 0,
        debug: bool = False
):
    """
    Train an FQL agent.

    Parameters
    ----------
    agent_cfg : FQLCfg
        Configuration for the FQL agent.
    trainer_cfg : TrainerCfg
        Configuration for the trainer.
    dataset_cfg : DatasetCfg
        Configuration for the dataset.
    env_name : str
        Name of the environment to train on.
    seed : int
        Random seed.
    debug : bool
        If True, run in debug mode.
    """
    _train(agent_cfg, trainer_cfg, dataset_cfg, env_name, seed, debug)


@app.command
def ifql(
        agent_cfg: IFQLCfg = IFQLCfg(),
        trainer_cfg: TrainerCfg = TrainerCfg(),
        dataset_cfg: DatasetCfg = DatasetCfg(),
        env_name: str = "cube-single-noisy-singletask-task1-v0",
        seed: int = 0,
        debug: bool = False
):
    """
    Train a IFQL agent.

    Parameters
    ----------
    agent_cfg : FQLCfg
        Configuration for the FQL agent.
    trainer_cfg : TrainerCfg
        Configuration for the trainer.
    dataset_cfg : DatasetCfg
        Configuration for the dataset.
    env_name : str
        Name of the environment to train on.
    seed : int
        Random seed.
    debug : bool
        If True, run in debug mode.
    """
    _train(agent_cfg, trainer_cfg, dataset_cfg, env_name, seed, debug)


@app.command
def dsrl(
        agent_cfg: DSRLCfg = DSRLCfg(),
        trainer_cfg: TrainerCfg = TrainerCfg(),
        dataset_cfg: DatasetCfg = DatasetCfg(),
        env_name: str = "cube-single-noisy-singletask-task1-v0",
        seed: int = 0,
        debug: bool = False
):
    """
    Train a DSRL agent.

    Parameters
    ----------
    agent_cfg : FQLCfg
        Configuration for the FQL agent.
    trainer_cfg : TrainerCfg
        Configuration for the trainer.
    dataset_cfg : DatasetCfg
        Configuration for the dataset.
    env_name : str
        Name of the environment to train on.
    seed : int
        Random seed.
    debug : bool
        If True, run in debug mode.
    """
    _train(agent_cfg, trainer_cfg, dataset_cfg, env_name, seed, debug)


@app.command
def reform(
        agent_cfg: ReFORMCfg = ReFORMCfg(),
        trainer_cfg: TrainerCfg = TrainerCfg(),
        dataset_cfg: DatasetCfg = DatasetCfg(),
        env_name: str = "cube-single-noisy-singletask-task1-v0",
        seed: int = 0,
        debug: bool = False
):
    """
    Train a ReFORM agent.

    Parameters
    ----------
    agent_cfg : FQLCfg
        Configuration for the FQL agent.
    trainer_cfg : TrainerCfg
        Configuration for the trainer.
    dataset_cfg : DatasetCfg
        Configuration for the dataset.
    env_name : str
        Name of the environment to train on.
    seed : int
        Random seed.
    debug : bool
        If True, run in debug mode.
    """
    _train(agent_cfg, trainer_cfg, dataset_cfg, env_name, seed, debug)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
