<div align="center">

# ReFORM

[![Conference](https://img.shields.io/badge/ICLR-Accepted-success)](https://mit-realm.github.io/reform/)

Jax official implementation of ICLR2026 paper: [Songyuan Zhang](https://syzhang092218-source.github.io), [Oswin So](https://oswinso.xyz/), [H. M. Sabbir Ahmad](https://sabbirahmad26.github.io/), [Eric Yang Yu](https://ericyangyu.github.io/), [Matthew Cleaveland](https://www.linkedin.com/in/matthew-cleaveland-4775abba/), [Mitchell Black](https://www.blackmitchell.com/), and [Chuchu Fan](https://chuchu.mit.edu): "[ReFORM: Reflected Flows for On-support Offline RL via Noise Manipulation](https://mit-realm.github.io/reform/)".

[Dependencies](#Dependencies) •
[Installation](#Installation) •
[Quickstart](#Quickstart) •
[Environments](#Environments) •
[Algorithms](#Algorithms) •
[Usage](#Usage) •
[Citation](#Citation)

</div>

<div align="center">
    <img src="./media/antmaze-large.gif" alt="antmaze-large" width="24.55%"/>
    <img src="./media/cube-single.gif" alt="cube-single" width="24.55%"/>
    <img src="./media/cube-double.gif" alt="cube-double" width="24.55%"/>
    <img src="./media/scene.gif" alt="scene" width="24.55%"/>
</div>

<div align="center">
    <img src="./media/reform.png" alt="ReFORM Framework" width="100%"/>
</div>

## Dependencies

We recommend to use [CONDA](https://www.anaconda.com/) to install the requirements:

```bash
conda create -n reform python=3.12
conda activate reform
```

Then install the dependencies:
```bash
pip install -r requirements.txt
```

## Installation

Install ReFORM: 

```bash
pip install -e .
```

## Quickstart

To train a model on the `cube-single-noisy-singletask-task1-v0` environment, run:

```bash
python scripts/train.py reform --env-name cube-single-noisy-singletask-task1-v0 --steps 3000000 --seed 0
```

To evaluate a model, run:

```bash
python scripts/test.py --path ./logs/cube-single-noisy-singletask-task1-v0/reform/seed0_xxxxxxxxxx
```

## Environments

We support the [OGBench](https://seohong.me/projects/ogbench/) benchmark environments. Since we are not doing goal-conditioned RL, make sure to use the `single-task` versions of the environments.

## Algorithms

We provide the following algorithms:

- `reform`: Our method: Reflected Flows for On-support Offline RL via Noise Manipulation.
- `fql`: [Flow Q-learning](https://seohong.me/projects/fql/).
- `ifql`: Flow version [IDQL](https://github.com/philippe-eecs/IDQL).
- `dsrl`: [Diffusion Steering via Reinforcement Learning ](https://diffusion-steering.github.io/).

## Usage

### Train

To train the `<algo>` algorithm on the `<env>` environment, run:

```bash
python scripts/train.py <algo> --env-name <env>
```

The training logs will be saved in `logs/<env>/<algo>/seed<seed>_<timestamp>`. Use the following command to check the available options:

```bash
python scripts/train.py <algo> -h
```

We provide the complete list of the exact command-line flags used to produce the main results of ReFORM in the paper.
ReFORM does not have environment-specific or dataset-specific hyperparameters, so **the same set of hyperparameters is used across all environments** except for the number of training steps. 
(`--q-agg` is a minor exception whose effect has not been well studied yet. We use the same option as the [FQL](https://seohong.me/projects/fql/) implementation).

<details>
<summary><b>Click to expand the full list of commands</b></summary>

Change `task1` to `task2`/`task3`/`task4`/`task5` to run on different tasks.

```bash
# ReFORM in antmaze-large environments with clean datasets.
python scripts/train.py reform --env-name antmaze-large-navigate-singletask-task1-v0 --q-agg min --steps 10000000 --seed 0
# ReFORM in antmaze-large environments with noisy datasets.
python scripts/train.py reform --env-name antmaze-large-explore-singletask-task1-v0 --q-agg min --steps 8000000 --seed 0
# ReFORM in cube-single environments with clean datasets.
python scripts/train.py reform --env-name cube-single-play-singletask-task1-v0 --steps 2000000 --seed 0
# ReFORM in cube-single environments with noisy datasets.
python scripts/train.py reform --env-name cube-single-noisy-singletask-task1-v0 --steps 3000000 --seed 0
# ReFORM in cube-double environments with clean datasets.
python scripts/train.py reform --env-name cube-double-play-singletask-task1-v0 --steps 2000000 --seed 0 
# ReFORM in cube-double environments with noisy datasets.
python scripts/train.py reform --env-name cube-double-noisy-singletask-task1-v0 --steps 1000000 --save-interval 50000 --seed 0 
# ReFORM in scene environments with clean datasets.
python scripts/train.py reform --env-name scene-play-singletask-task1-v0 --steps 2000000 --seed 0
# ReFORM in scene environments with noisy datasets.
python scripts/train.py reform --env-name scene-noisy-singletask-task1-v0 --steps 1000000 --seed 0
# ReFORM in visual-cube-single environments with clean datasets.
python scripts/train.py reform --env-name visual-cube-single-play-singletask-task1-v0 --steps 1000000 --encoder impala_small --p_aug 0.5 --frame_stack 3 --seed 0
# ReFORM in visual-cube-single environments with noisy datasets.
python scripts/train.py reform --env-name visual-cube-single-noisy-singletask-task1-v0 --steps 1000000 --encoder impala_small --p_aug 0.5 --frame_stack 3 --seed 0
```

</details>

### Test

To test the learned model, use:

```bash
python scripts/test.py --path <path-to-log>
```

This should report the mean reward and the safety rate of the learned model. Also, it will generate videos of the learned model in `<path-to-log>/videos`. Use the following flag to check the available options:

```bash
python scripts/test.py -h
```

## Acknowledgements

This codebase is built upon the [FQL](https://github.com/seohongpark/fql/) implementation.

## Citation

```
@inproceedings{zhang2026reform,
      title={Re{FORM}: Reflected Flows for On-support Offline {RL} via Noise Manipulation},
      author={Zhang, Songyuan and So, Oswin and Ahmad, H M Sabbir and Yu, Eric Yang and Cleaveland, Matthew and Black, Mitchell and Fan, Chuchu},
      booktitle={The Fourteenth International Conference on Learning Representations},
      year={2026},
}
```