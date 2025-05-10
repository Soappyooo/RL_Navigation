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

    seed_all(0)

    n_envs = 32  # At least 24GB RAM
    # n_envs = 1
    n_envs_eval = 4
    seq_len = 4
    episode_length = 1024
    time_scale = 5
    # time_scale = 1
    obs_interval = 5
    use_amp = True
    env = make_vec_env(
        EpMineEnv,
        env_kwargs=dict(
            time_scale=time_scale,
            max_episode_steps=episode_length,
            only_image=False,
            only_state=False,
            history_length=seq_len,
            obs_interval=obs_interval,
        ),
        n_envs=n_envs,
        seed=0,
        vec_env_cls=SubprocVecEnv,
    )

    # model = PPO(
    #     "MultiInputPolicy",
    #     env,
    #     verbose=1,
    #     ent_coef=0.01,
    #     tensorboard_log="./logs/tensorboard",
    #     seed=0,
    # )
    model = CustomPPO(
        NavActorCriticPolicy,
        env,
        policy_kwargs=dict(
            backbone_name="efficientnet",
            encoder_name="lstm",
            from_pretrained=True,
            pretrained_backbone_path=Path("./checkpoints/pretrained/vint_pretrained_efficientnet.pth"),
            hidden_dim=256,
            head_hidden_dims=(128, 64),
            max_seq_len=seq_len,
        ),
        n_steps=8192 // n_envs,
        batch_size=512,  # 512 for 16 GB VRAM
        n_epochs=5,
        learning_rate=1e-4,
        ent_coef=0.0005,
        vf_coef=0.5,
        pose_coef=0.2,
        clip_range=0.2,
        verbose=1,
        seed=0,
        use_amp=use_amp,
        tensorboard_log="./logs/tensorboard",
    )

    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=2e5 // n_envs, save_path="./checkpoints/training", name_prefix="ppo_model"
    )

    # Create eval environment
    eval_env = make_vec_env(
        EpMineEnv,
        env_kwargs=dict(
            time_scale=time_scale,
            max_episode_steps=episode_length,
            only_image=True,
            only_state=False,
            history_length=seq_len,
            obs_interval=obs_interval,
        ),
        n_envs=n_envs_eval,
        seed=42,
        vec_env_cls=SubprocVecEnv,
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path="./checkpoints/best",
        eval_freq=1e5 // n_envs,
        n_eval_episodes=20,  # should be multiple of n_envs
        deterministic=True,
    )

    model.learn(total_timesteps=1e6, callback=[checkpoint_callback, eval_callback])


if __name__ == "__main__":
    main()
