from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.base_env import ActionTuple, DecisionSteps
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Deque
import time
import cv2 as cv
import gym
import socket
from collections import deque
from .config import UNITY_ENV_PATH
from models.backbones import VisualBackbone

# Constants for Unity environment configuration
TEAM_NAME = "ControlEP?team=0"
AGENT_ID = 0
IMAGE_BACKBONE_MODE = "vint"

if IMAGE_BACKBONE_MODE == "vint":
    DEFAULT_IMAGE_SIZE = (3, 64, 85)  # CHW format after preprocessing
else:
    DEFAULT_IMAGE_SIZE = (3, 128, 128)
DEFAULT_STATE_SIZE = 3 + 9  # 3 for translation, 9 for rotation matrix


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
        elif result == 10035:
            print(f"Port {port} is available on {host}")
            return True
        else:
            print(f"Port check failed with error code: {result}")
            return False


def quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.

    Args:
        quat (np.ndarray): Quaternion [x, y, z, w]

    Returns:
        np.ndarray: Rotation matrix of shape (3, 3)
    """
    x, y, z, w = quat
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )


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
        time_scale: float = 100,
        max_episode_steps: int = 1000,
        only_image: bool = False,
        only_state: bool = False,
        no_graphics: bool = False,
        history_length: int = 8,
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
            history_length (int): Number of frames to stack in history
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
        self.history_length = history_length

        # Observation type flags
        self.only_image = only_image
        self.only_state = only_state

        # Episode state
        self.step_count = 0
        self.last_distance = 0.0
        self.current_results = None
        self.gripper_state = 0

        # Initialize observation history
        self.image_history: Deque = deque(maxlen=history_length)
        self.state_history: Deque = deque(maxlen=history_length)

    @property
    def observation_space(self) -> gym.spaces.Dict:
        """Define the observation space structure."""
        # Image space now includes history dimension
        image_shape = (self.history_length,) + DEFAULT_IMAGE_SIZE
        state_shape = (self.history_length, DEFAULT_STATE_SIZE)

        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(low=0, high=1.0, shape=image_shape, dtype=np.float32),
                "state": gym.spaces.Box(low=-np.Inf, high=np.Inf, shape=state_shape, dtype=np.float32),
            }
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        """Define the action space for robot control."""
        return gym.spaces.Box(
            low=np.array([-10.0, -10.0, -3.0]), high=np.array([10.0, 10.0, 3.0]), shape=(3,), dtype=np.float32
        )
        
    def seed(self, seed: Optional[int] = None) -> None:
        """
        Set the random seed for the environment.

        Args:
            seed (int, optional): Random seed for reproducibility
        """
        if seed is not None:
            self.seed_value = seed
        if self.env is not None:
            self.env.set_seed(seed)
        self.work_id = seed if seed is not None else self.work_id

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
            timeout_wait=10,
        )

    def decode_observation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode the observation from Unity environment results.

        Args:
            results: Raw results from Unity environment

        Returns:
            Processed observation dictionary with history
        """
        obs = results.obs
        # Process image observation (uint8 -> float32, normalized)
        image = np.array(obs[0][AGENT_ID] * 255, dtype=np.uint8)

        image = VisualBackbone.preprocess_image(image, mode=IMAGE_BACKBONE_MODE)  # Returns CHW format

        # Extract state information
        state = obs[1][AGENT_ID]
        self.gripper_state = state[8]  # Update gripper state
        state_with_rotmat = np.concatenate((state[4:7], quat_to_rotmat(state[0:4]).flatten()), axis=0)  # (3 + 9)

        # Add current observation to history
        self.image_history.append(image)
        self.state_history.append(state_with_rotmat)

        # If history is not full, repeat the first observation
        while len(self.image_history) < self.history_length:
            self.image_history.append(image)
            self.state_history.append(state_with_rotmat)

        # Stack history into arrays
        image_stack = np.stack(list(self.image_history), axis=0)
        state_stack = np.stack(list(self.state_history), axis=0)

        # Return observation dict with history
        if self.only_image:
            return {"image": image_stack, "state": np.zeros_like(state_stack)}
        elif self.only_state:
            return {"image": np.zeros_like(image_stack), "state": state_stack}
        return {"image": image_stack, "state": state_stack}

    def get_robot_pose(self, results: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract robot position and rotation from results."""
        obs = results.obs
        position = obs[1][AGENT_ID][4:7]
        rotation = quat_to_rotmat(obs[1][AGENT_ID][0:4])
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

        # Clear history
        self.image_history.clear()
        self.state_history.clear()
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

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, dict]:
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

        # No need to include robot pose in info since it's in observation
        return obs, reward, done, {}

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
        else:
            self.current_results = decision_steps
        obs = self.decode_observation(self.current_results)
        reward = self.calculate_reward(self.current_results)
        # robot_position = self.get_robot_pose(self.current_results)[0]
        robot_position, robot_rotation = self.get_robot_pose(self.current_results)

        # Check episode length limit
        if self.step_count >= self.max_episode_steps:
            done = True

        info["robot_position"] = robot_position
        info["robot_rotation"] = robot_rotation
        return obs, reward, done, info


def main():
    """Example usage of the environment."""
    env = EpMineEnv(port=3000)
    obs = env.reset()
    done = False
    step = 0

    last_time = time.perf_counter()
    while not done:
        current_time = time.perf_counter()
        elapsed_time = current_time - last_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        last_time = current_time

        # action = env.action_space.sample()
        action = np.array([0.0, 0.0, 0.1], dtype=np.float32)
        obs, reward, done, info = env.step(action)
        position = info["robot_position"]
        rotation = info["robot_rotation"]
        print(f"Step: {step}, Action: {action}, Reward: {reward}, Position: {position}, Rotation: {rotation}")
        step += 1


if __name__ == "__main__":
    main()
