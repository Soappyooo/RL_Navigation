import gym
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from envs.SingleAgent.mine_toy import EpMineEnv
from algorithms.nav_policy_wrapper import NavActorCriticPolicy
from algorithms.custom_ppo import CustomPPO
from pathlib import Path
import torch
import random


def seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():

    # check current working directory is EpMineEnv-main,
    # because we save things relative to this directory
    if Path.cwd().name != "EpMineEnv-main":
        raise ValueError(
            f"Please run this script from the EpMineEnv-main directory." f" Current directory: {Path.cwd()}"
        )

    seed = 0  # seed not working for the env
    n_envs = 32  # 32 for At least 24GB RAM
    seq_len = 1  # sequence length (num frames) for observation, also determines model input shape
    episode_length = 2048  # max episode length. Actual episode length decreases with training
    time_scale = 5  # time scale for the environment, larger values make the environment runs faster but renders slower
    obs_interval = 1  # interval between observations, e.g. 1 means every frame, 2 means every other frame
    use_amp = False  # use automatic mixed precision for training

    seed_all(seed)
    env = make_vec_env(
        EpMineEnv,
        env_kwargs=dict(
            time_scale=time_scale,
            max_episode_steps=episode_length,
            only_image=False,  # True for only image, which will set state to all zero
            only_state=False,
            history_length=seq_len,
            obs_interval=obs_interval,
            image_preprocess_mode="simple",
        ),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
    )

    model = CustomPPO(
        NavActorCriticPolicy,
        env,
        policy_kwargs=dict(  # arguments passed to NavActorCriticPolicy.__init__()
            backbone_name="simple",
            encoder_name="identity",
            pose_auxiliary_mode="concatenate",
            hidden_dim=512,
            head_hidden_dims=(256, 128),
            seq_len=seq_len,
            optimizer_kwargs=dict(weight_decay=0),
        ),
        n_steps=8192 // n_envs,  # before each update, envs collect a total number of n_steps*n_envs steps
        batch_size=2048,  # batch size for training
        n_epochs=10,  # number of epochs for training
        learning_rate=3e-4,
        ent_coef=3e-5,  # entropy coefficient
        vf_coef=0.5,  # value function coefficient
        pose_coef=0.02,  # pose auxiliary loss coefficient, big value may be unstable
        clip_range=0.2,
        gae_lambda=0.95,
        verbose=1,
        seed=seed,
        use_amp=use_amp,
        tensorboard_log="./logs/tensorboard",
    )
    total_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")

    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=1e5 // n_envs, save_path="./checkpoints/training", name_prefix="ppo_model"
    )

    model.learn(total_timesteps=1e6, callback=[checkpoint_callback], reset_num_timesteps=True)


if __name__ == "__main__":
    main()
