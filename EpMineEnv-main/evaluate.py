import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from envs.SingleAgent.mine_toy import EpMineEnv
from algorithms.nav_policy_wrapper import NavActorCriticPolicy
from algorithms.custom_ppo import CustomPPO
import argparse

def evaluate_model(model_path: str, n_episodes: int = 10):
    # Create environment
    env = make_vec_env(
        EpMineEnv,
        env_kwargs=dict(
            time_scale=1,
            max_episode_steps=1000,
            only_image=True,
            only_state=False,
            history_length=8,
        ),
        n_envs=1,
        seed=42,
    )
    
    # Load the trained model
    model = PPO.load(model_path, env=env)
    # Uncomment below for CustomPPO model
    # model = CustomPPO.load(model_path, env=env)
    
    rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1
                
        rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1} reward: {episode_reward:.2f}, length: {episode_length}")
    
    print("\nEvaluation Results:")
    print(f"Mean reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model")
    parser.add_argument("--model-path", type=str, required=True, 
                      help="Path to the saved model")
    parser.add_argument("--episodes", type=int, default=10,
                      help="Number of episodes to evaluate")
    
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.episodes)