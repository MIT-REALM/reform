import cyclopts
import ipdb
import os
import numpy as np
import yaml
import dataclasses
import jax.random as jr
import cv2
import datetime

from reform.trainer.utils import restore_agent, test_actor, supply_rng
from reform.agents import agents, agent_cfgs
from reform.env import make_env_and_datasets


app = cyclopts.App()


@app.default
def test(
        path: str,
        step: int = None,
        epi: int = 10,
        seed: int = 0,
        no_video: bool = False,
        debug: bool = False
):
    """
    Test a trained agent in the specified environment and generate videos.

    Parameters
    ----------
    path : str
        Path to the directory containing the trained model and config.yaml.
    step : int | None
        Specific training step to load the model from. If None, loads the latest model.
    epi : int
        Number of episodes to test.
    seed : int
        Random seed for testing.
    no_video : bool
        If True, do not generate videos.
    debug : bool
        If True, run in debug mode.
    """
    # Set up environment variables and seed.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if debug:
        os.environ["JAX_DISABLE_JIT"] = "True"
    global_rng = np.random.default_rng(seed)
    np.random.seed(global_rng.integers(2**31))

    # Load config.
    with open(os.path.join(path, 'config.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)

    # Create environment.
    env, train_dataset, _ = make_env_and_datasets(
        config['env_name'], render_mode="rgb_array", width=1080, height=720
    )

    # Load model.
    model_path = os.path.join(path, 'models')
    if step is not None:
        load_step = step
    else:
        models = os.listdir(model_path)
        load_step = max([int(m.split('_')[-1].split('.')[0]) for m in models if m.startswith('params_')])

    agent_class = agents[config['agent_name']]
    agent_cfg = agent_cfgs[config['agent_name']]()
    valid_keys = [f.name for f in dataclasses.fields(agent_cfg)]
    agent_cfg = dataclasses.replace(agent_cfg, **{k: v for k, v in config.items() if k in valid_keys})
    agent = agent_class.create(
        global_rng.integers(2**31),
        train_dataset['observations'][0][None],
        train_dataset['actions'][0][None],
        agent_cfg,
    )
    agent = restore_agent(agent, model_path, load_step)
    print('Loaded model from:', model_path)

    # Test the agent in the environment.
    key = jr.PRNGKey(global_rng.integers(2**31))
    actor_fn = supply_rng(agent.sample_actions, key=key)
    trajs, frames = test_actor(
        actor_fn,
        env,
        n_episodes=epi,
        rng=global_rng,
        render=not no_video,
        verbose=True,
    )
    reward_mean = np.mean([np.sum(traj['reward']) for traj in trajs])
    success_mean = np.mean([traj['info'][-1]['success'] for traj in trajs])
    print(f'Test results over {epi} episodes -- '
          f'Mean Reward: {reward_mean:.2f}, Mean Success: {success_mean:.2f}')

    # Generate video.
    if not no_video:
        videos_dir = os.path.join(path, 'videos', f'step_{load_step}')
        os.makedirs(videos_dir, exist_ok=True)
        stamp_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        for i in range(len(frames)):
            episode_frames = frames[i]
            episode_name = (f'epi{i}_reward{int(np.sum(trajs[i]["reward"]))}_'
                            f'success{int(trajs[i]["info"][-1]["success"])}_{stamp_str}.mp4')
            episode_path = os.path.join(videos_dir, episode_name)
            height, width, _ = episode_frames[0].shape
            video_writer = cv2.VideoWriter(episode_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
            for frame in episode_frames:
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            video_writer.release()
        print('Saved video to: ', videos_dir)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
