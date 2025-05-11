import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from envs.SingleAgent.mine_toy import EpMineEnv
from algorithms.nav_policy_wrapper import NavActorCriticPolicy
from algorithms.custom_ppo import CustomPPO
import argparse


def evaluate_model(
    model_path: str, n_episodes: int = 10, n_envs: int = 1, deterministic: bool = True, time_scale: int = 1
):
    # Create environment
    env = make_vec_env(
        EpMineEnv,
        env_kwargs=dict(
            time_scale=time_scale,
            max_episode_steps=512,
            only_image=True,
            only_state=False,
            history_length=8,
            render_size=(200, 100),
        ),
        n_envs=n_envs,
        seed=42,
        vec_env_cls=SubprocVecEnv,
    )

    # Load the trained model
    model = CustomPPO.load(model_path, env=env)

    episode_rewards, episode_lengths = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_episodes,
        deterministic=deterministic,
        render=False,
        return_episode_rewards=True,
    )

    # Print evaluation results
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean episode length: {mean_length:.2f} +/- {std_length:.2f}")

    env.close()


if __name__ == "__main__":
    default_model_path = (
        "/app/EpMineEnv-main/checkpoints/training/ppo_model_1000000_steps.zip"  # Change this to your model path
    )
    default_n_envs = 8
    default_episodes = 10 * default_n_envs
    default_time_scale = 5
    default_deterministic = False

    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model")
    parser.add_argument("--model-path", type=str, default=default_model_path, help="Path to the saved model")
    parser.add_argument("--episodes", type=int, default=default_episodes, help="Number of episodes to evaluate")
    parser.add_argument("--n-envs", type=int, default=default_n_envs, help="Number of parallel environments")
    parser.add_argument("--time-scale", type=int, default=default_time_scale, help="Time scale for the environment")
    parser.add_argument(
        "--deterministic", action="store_true", default=default_deterministic, help="Use deterministic actions"
    )

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        n_episodes=args.episodes,
        n_envs=args.n_envs,
        deterministic=args.deterministic,
        time_scale=args.time_scale,
    )
