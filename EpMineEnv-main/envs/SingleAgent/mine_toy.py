from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.base_env import ActionTuple, DecisionSteps
import numpy as np
from typing import Optional, Tuple, Dict, Any
import time
import cv2 as cv
import gym
import socket
from .config import UNITY_ENV_PATH

# Constants for Unity environment configuration
TEAM_NAME = "ControlEP?team=0"
AGENT_ID = 0
DEFAULT_IMAGE_SIZE = (128, 128, 3)
DEFAULT_STATE_SIZE = 7


def check_port_availability(port: int, host: str = "127.0.0.1") -> bool:
    """
    Check if a port is available on the specified host.

    Args:
        port (int): Port number to check
        host (str): Host address to check. Defaults to localhost

    Returns:
        bool: True if port is available, False if in use
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        if result == 0:
            print(f"Port {port} is in use on {host}")
            return False
        elif result == 111:
            print(f"Port {port} is available on {host}")
            return True
        else:
            print(f"Port check failed with error code: {result}")
            return False


class EpMineEnv(gym.Env):
    """
    A custom Gym environment for robotic mineral collection task using Unity ML-Agents.

    This environment simulates a robot that needs to navigate to and collect minerals
    in a given space. The robot can move in 3D space and control its arm for collection.
    """

    def __init__(
        self,
        file_name: str = UNITY_ENV_PATH,  # Use the config path by default
        port: Optional[int] = 2000,
        seed: int = 0,
        work_id: int = 0,
        time_scale: float = 10,
        max_episode_steps: int = 1000,
        only_image: bool = True,
        only_state: bool = False,
        no_graphics: bool = False,
    ):
        """
        Initialize the environment with given parameters.

        Args:
            file_name (str): Path to the Unity executable, defaults to platform-specific path
            port (int, optional): Base port for Unity communication
            seed (int): Random seed for reproducibility
            work_id (int): Worker ID for parallel environments
            time_scale (float): Unity time scale factor
            max_episode_steps (int): Maximum steps per episode
            only_image (bool): If True, observation space only includes image
            only_state (bool): If True, observation space only includes state vector
            no_graphics (bool): If True, runs Unity environment without graphics
        """
        # Initialize Unity channels
        self.engine_config_channel = EngineConfigurationChannel()
        self.engine_config_channel.set_configuration_parameters(
            width=200, height=100, time_scale=time_scale  # Large time_scale result in low FPS
        )
        self.env_param_channel = EnvironmentParametersChannel()

        # Environment setup
        self.env = None
        self.port = port
        self.work_id = work_id
        self.env_file_name = file_name
        self.seed_value = seed
        self.no_graphics = no_graphics
        self.max_episode_steps = max_episode_steps

        # Observation type flags
        self.only_image = only_image
        self.only_state = only_state

        # Episode state
        self.step_count = 0
        self.last_distance = 0.0
        self.current_results = None
        self.gripper_state = 0

    @property
    def observation_space(self) -> gym.spaces.Space:
        """Define the observation space structure based on configuration."""
        state_space = gym.spaces.Box(low=-np.Inf, high=np.Inf, shape=(DEFAULT_STATE_SIZE,), dtype=np.float32)

        if self.only_image:
            return gym.spaces.Box(low=0, high=255, shape=DEFAULT_IMAGE_SIZE, dtype=np.uint8)
        elif self.only_state:
            return state_space

        return gym.spaces.Dict(
            {"image": gym.spaces.Box(low=0, high=255, shape=DEFAULT_IMAGE_SIZE, dtype=np.uint8), "state": state_space}
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        """Define the action space for robot control."""
        return gym.spaces.Box(
            low=np.array([-10.0, -10.0, -3.0]), high=np.array([10.0, 10.0, 3.0]), shape=(3,), dtype=np.float32
        )

    def initialize_environment(self, seed: Optional[int] = None) -> None:
        """
        Initialize or reinitialize the Unity environment.

        Args:
            seed (int, optional): Random seed for the environment
        """
        if self.env is not None:
            self.env.close()

        worker_id = self.work_id
        while not check_port_availability(self.port + worker_id):
            worker_id += 1

        self.env = UnityEnvironment(
            file_name=self.env_file_name,
            base_port=self.port,
            seed=seed or self.seed_value,
            worker_id=worker_id,
            side_channels=[self.env_param_channel, self.engine_config_channel],
            no_graphics=self.no_graphics,
        )

    def decode_observation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode the observation from Unity environment results.

        Args:
            results: Raw results from Unity environment

        Returns:
            Processed observation dictionary
        """
        obs = results.obs
        # Process image observation
        image = cv.cvtColor(np.array(obs[0][AGENT_ID] * 255, dtype=np.uint8), cv.COLOR_RGB2BGR)

        # Extract state information
        state = obs[1][AGENT_ID]
        self.gripper_state = state[8]  # Update gripper state

        if self.only_image:
            return image
        elif self.only_state:
            return np.array(state[:7])

        return {"image": image, "state": state}

    def get_robot_pose(self, results: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract robot position and rotation from results."""
        obs = results.obs
        position = obs[1][AGENT_ID][4:7]
        rotation = obs[1][AGENT_ID][0:4]
        return position, rotation

    def get_mineral_pose(self, results: Dict[str, Any]) -> np.ndarray:
        """Extract mineral position from results."""
        return results.obs[1][AGENT_ID][10:13]

    def calculate_distance_to_mineral(self, results: Dict[str, Any]) -> float:
        """Calculate distance between robot and mineral."""
        robot_pos = self.get_robot_pose(results)[0]
        return np.sqrt(robot_pos[0] ** 2 + robot_pos[2] ** 2)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Any:
        """Reset the environment to initial state."""
        if seed is not None:
            self.seed_value = seed
        if self.env is None:
            self.initialize_environment(self.seed_value)

        self.step_count = 0
        self.env.reset()
        obs, _, _, _ = self._step()
        self.last_distance = self.calculate_distance_to_mineral(self.current_results)
        return obs

    def calculate_reward(self, results: Dict[str, Any]) -> float:
        """
        Calculate the reward based on current state and results.

        Includes both sparse reward from Unity and dense reward based on distance change.
        """
        sparse_reward = results.reward[AGENT_ID]
        current_dist = self.calculate_distance_to_mineral(results)
        distance_reward = self.last_distance - current_dist
        self.last_distance = current_dist

        return sparse_reward + distance_reward

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, dict]:
        """
        Execute one environment step with given action.

        Args:
            action: Array of 3 values [vy, vx, vw] for robot control

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Extend action with fixed arm angle and gripper control
        full_action = np.array([action[0], action[1], action[2], 10.0, 1.0], dtype=np.float32)

        action_tuple = ActionTuple(np.array([full_action]))
        obs, reward, done, info = self._step(action_tuple)
        self.step_count += 1

        return obs, reward, done, info

    def _step(self, action: Optional[ActionTuple] = None) -> Tuple[Any, float, bool, dict]:
        """Internal step function that handles Unity environment interaction."""
        if action is not None:
            self.env.set_action_for_agent(TEAM_NAME, AGENT_ID, action)
        self.env.step()

        decision_steps, terminal_steps = self.env.get_steps(TEAM_NAME)

        # Initialize return values
        done = False
        info = {}

        # Handle terminal state
        if len(terminal_steps) != 0:
            done = True
            self.current_results = terminal_steps
            obs = self.decode_observation(terminal_steps)
            reward = self.calculate_reward(terminal_steps)
            robot_position = self.get_robot_pose(terminal_steps)[0]
        else:
            self.current_results = decision_steps
            obs = self.decode_observation(decision_steps)
            reward = self.calculate_reward(decision_steps)
            robot_position = self.get_robot_pose(decision_steps)[0]

        # Check episode length limit
        if self.step_count >= self.max_episode_steps:
            done = True

        info["robot_position"] = robot_position
        return obs, reward, done, info


def main():
    """Example usage of the environment."""
    env = EpMineEnv(port=3000)
    obs = env.reset()
    done = False
    step = 0

    while not done:
        print(f"Step time: {time.time()}")
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        position = info["robot_position"]

        # Save observation image
        cv.imwrite(f"images/{step}-({position[0]:.2f}, {position[2]:.2f}).png", obs)
        print("-" * 40)
        step += 1


if __name__ == "__main__":
    main()
