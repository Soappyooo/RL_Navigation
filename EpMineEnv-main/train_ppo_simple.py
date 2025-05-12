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


def main():

    # check current working directory is EpMineEnv-main,
    # because we save things relative to this directory
    if Path.cwd().name != "EpMineEnv-main":
        raise ValueError(
            f"Please run this script from the EpMineEnv-main directory." f" Current directory: {Path.cwd()}"
        )

    seed = 0
    n_envs = 32  # 32 for At least 24GB RAM
    n_envs_eval = 2
    seq_len = 1
    episode_length = 2048
    time_scale = 5
    obs_interval = 1
    use_amp = True

    seed_all(seed)
    env = make_vec_env(
        EpMineEnv,
        env_kwargs=dict(
            time_scale=time_scale,
            max_episode_steps=episode_length,
            only_image=True,
            only_state=False,
            history_length=seq_len,
            obs_interval=obs_interval,
            image_preprocess_mode="simple",
        ),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
    )

    # TODO: weight decay 1e-3 (https://openreview.net/forum?id=m9Jfdz4ymO)
    # TODO: seperate pose prediction head
    # TODO: lr 3e-4, n_epochs 3, gae_lambda 0.9, ent_coef 3e-5

    model = CustomPPO(
        NavActorCriticPolicy,
        env,
        policy_kwargs=dict(
            backbone_name="simple",
            encoder_name="identity",
            hidden_dim=512,
            head_hidden_dims=(256, 128),
            max_seq_len=seq_len,
            optimizer_kwargs=dict(weight_decay=0),
        ),
        # n_steps=8192 // n_envs,
        n_steps=int(episode_length * 0.125),  # every env collect at least n episode before update
        batch_size=512,  # 512 and seq_len=4 for 16 GB VRAM
        n_epochs=10,
        learning_rate=3e-4,
        ent_coef=3e-5,
        vf_coef=0.5,
        pose_coef=0,
        clip_range=0.2,
        gae_lambda=0.95,
        verbose=1,
        seed=seed,
        use_amp=use_amp,
        tensorboard_log="./logs/tensorboard",
    )
    total_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")  # 1.6M

    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=1e5 // n_envs, save_path="./checkpoints/training", name_prefix="ppo_model"
    )

    model.learn(total_timesteps=1e6, callback=[checkpoint_callback], reset_num_timesteps=True)


if __name__ == "__main__":
    main()
