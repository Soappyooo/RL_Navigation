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


def main():

    # check current working directory is EpMineEnv-main,
    # because we save things relative to this directory
    if Path.cwd().name != "EpMineEnv-main":
        raise ValueError(
            f"Please run this script from the EpMineEnv-main directory." f" Current directory: {Path.cwd()}"
        )

    n_envs = 4
    seq_len = 8
    use_amp = True
    env = make_vec_env(
        EpMineEnv,
        env_kwargs=dict(
            time_scale=100,
            max_episode_steps=1024,
            only_image=True,
            only_state=False,
            history_length=seq_len,
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
        batch_size=288,  # 288 for 16 GB VRAM
        n_epochs=5,
        learning_rate=1e-3,
        ent_coef=0.01,
        pose_coef=0.1,
        clip_range=0.2,
        verbose=1,
        seed=0,
        use_amp=use_amp,
        tensorboard_log="./logs/tensorboard",
    )

    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=1e5 // n_envs, save_path="./checkpoints/training", name_prefix="ppo_model"
    )

    # Create eval environment
    eval_env = make_vec_env(
        EpMineEnv,
        env_kwargs=dict(
            time_scale=100,
            max_episode_steps=1024,
            only_image=True,
            only_state=False,
            history_length=seq_len,
        ),
        n_envs=1,
        seed=42,
        vec_env_cls=SubprocVecEnv,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/best",
        eval_freq=5e4 // n_envs,
        n_eval_episodes=5,
        deterministic=True,
    )

    model.learn(total_timesteps=1e6, callback=[checkpoint_callback, eval_callback])


if __name__ == "__main__":
    main()
