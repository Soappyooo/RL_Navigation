import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from envs.SingleAgent.mine_toy import EpMineEnv
from algorithms.nav_policy_wrapper import NavActorCriticPolicy
from algorithms.custom_ppo import CustomPPO
from pathlib import Path

if __name__ == "__main__":
    n_envs = 4  # Number of Environments to run in parallel
    # Create the vectorized environment
    # env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    env = make_vec_env(
        EpMineEnv,
        env_kwargs=dict(
            time_scale=100,
            max_episode_steps=1000,
            only_image=True,
            only_state=False,
            history_length=8,
        ),
        n_envs=n_envs,
        seed=0,
        vec_env_cls=SubprocVecEnv,
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        ent_coef=0.01,
        tensorboard_log="/app/EpMineEnv-main/logs/tensorboard",
        seed=0,
    )
    # model = CustomPPO(
    #     NavActorCriticPolicy,
    #     env,
    #     policy_kwargs=dict(
    #         backbone_name="efficientnet",
    #         encoder_name="lstm",
    #         from_pretrained=True,
    #         pretrained_backbone_path=Path("/app/EpMineEnv-main/checkpoints/vint_pretrained_efficientnet.pth"),
    #     ),
    #     n_steps=2048,
    #     batch_size=64,  # 64 for 8 GB VRAM
    #     n_epochs=10,
    #     learning_rate=3e-4,
    #     ent_coef=0.01,
    #     pose_coef=0.1,
    #     clip_range=0.2,
    #     verbose=1,
    #     seed=0,
    #     tensorboard_log="/app/EpMineEnv-main/logs/tensorboard",
    # )
    model.learn(total_timesteps=1e6)

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
