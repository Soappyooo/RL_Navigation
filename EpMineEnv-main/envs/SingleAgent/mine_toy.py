from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.base_env import ActionTuple, DecisionSteps
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Deque
import time
import cv2
import gym
import socket
from collections import deque

from pynput import keyboard
import atexit

from .config import UNITY_ENV_PATH

# Constants for Unity environment configuration
TEAM_NAME = "ControlEP?team=0"
AGENT_ID = 0
DEBUG = False  # Set to True for debugging, run play.py to test the environment

# Global key state dictionary to track pressed keys
key_states = {
    "w": False,  # Forward
    "s": False,  # Backward
    "a": False,  # Left
    "d": False,  # Right
    "q": False,  # Rotate left
    "e": False,  # Rotate right
}


# Key press and release callback functions
def on_press(key):
    """
    Callback function when a key is pressed
    """
    try:
        k = key.char.lower()  # Convert to lowercase
        if k in key_states:
            key_states[k] = True
    except AttributeError:
        # Special keys that don't have a char attribute
        pass


def on_release(key):
    """
    Callback function when a key is released
    """
    try:
        k = key.char.lower()
        if k in key_states:
            key_states[k] = False
    except AttributeError:
        pass


def get_action_from_keyboard() -> Tuple[float, float, float]:
    """
    Read keyboard input and convert to robot actions without waiting.
    Uses pynput to detect pressed keys globally, doesn't require an active window.

    Controls:
    - W/S: Forward/Backward movement (action[0])
    - A/D: Left/Right movement (action[1])
    - Q/E: Rotation (action[2])

    Returns:
        Tuple[float, float, float]: Action values for [forward/backward, left/right, rotation]
    """
    # Default action (no movement)
    action_y = 0.0  # Forward/Backward
    action_x = 0.0  # Left/Right
    action_w = 0.0  # Rotation

    # Movement speed multipliers
    move_speed = 5.0
    rotation_speed = 1.0

    # Check global key states
    if key_states["w"]:
        action_y = move_speed
    elif key_states["s"]:
        action_y = -move_speed

    if key_states["a"]:
        action_x = -move_speed
    elif key_states["d"]:
        action_x = move_speed

    if key_states["q"]:
        action_w = -rotation_speed
    elif key_states["e"]:
        action_w = rotation_speed

    return action_y, action_x, action_w


def exit_handler():
    """
    Stop the keyboard listener when the program exits
    """
    if listener.is_alive():
        listener.stop()


# Setup keyboard listener
if DEBUG:
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    atexit.register(exit_handler)


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


def preprocess_image(image: np.ndarray, mode: str = None) -> np.ndarray:
    """
    Preprocess the input image according to the checkpoint. E.g. `mode="vint"` result in
    resizing the image to 85*64 and normalizing it with mean and std.
    Args:
        image (np.ndarray): Input image. H*W*3 in RGB format, uint8 (0-255).
        mode (str, optional): Mode for preprocessing. Defaults to None.
    Returns:
        np.ndarray: Preprocessed image tensor of shape [3, H', W'].
    """
    # Check input shape and type
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy array.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be of shape H*W*3.")
    if image.dtype != np.uint8:
        raise ValueError("Input image must be of type uint8.")

    if mode == "vint":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = cv2.resize(image, (85, 64))
        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        return image

    elif mode == "simple":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = cv2.resize(image, (128, 128))
        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        return image

    else:
        raise NotImplementedError(f"Preprocessing for mode '{mode}' is not implemented.")


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
        image_preprocess_mode: str = "vint",
        obs_interval: int = 1,
        render_size: Tuple[int, int] = (200, 100),
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
            image_preprocess_mode (str): Preprocessing mode for images
            obs_interval (int): Interval between observations in the history (default: 1)
            render_size (Tuple[int, int]): Size of the rendered image
        """
        # Initialize Unity channels
        self.engine_config_channel = EngineConfigurationChannel()
        self.engine_config_channel.set_configuration_parameters(
            width=render_size[0], height=render_size[1], time_scale=time_scale  # Large time_scale result in low FPS
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
        self.image_preprocess_mode = image_preprocess_mode
        self.obs_interval = max(1, obs_interval)  # Ensure interval is at least 1

        # Set the image and state sizes
        self.state_size = 2
        if self.image_preprocess_mode == "vint":
            self.image_size = (3, 64, 85)
        elif self.image_preprocess_mode == "simple":
            self.image_size = (3, 128, 128)
        else:
            raise NotImplementedError(f"Preprocessing for mode '{self.image_preprocess_mode}' is not implemented.")

        # Observation type flags
        self.only_image = only_image
        self.only_state = only_state

        # Episode state
        self.step_count = 0
        self.last_distance = 0.0
        self.last_nearby = 0
        self.current_results = None
        self.gripper_state = 0

        # Initialize observation history
        # We need to store more observations to accommodate the interval
        max_buffer_size = history_length * self.obs_interval
        self.image_buffer: Deque = deque(maxlen=max_buffer_size)
        self.state_buffer: Deque = deque(maxlen=max_buffer_size)

        # Time counter
        self.time_counter = time.perf_counter()

    @property
    def observation_space(self) -> gym.spaces.Dict:
        """Define the observation space structure."""
        # Image space now includes history dimension
        image_shape = (self.history_length,) + self.image_size
        state_shape = (self.history_length, self.state_size)

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
            self.env.set_seed(self.seed_value)
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
            timeout_wait=60,
        )

    def decode_observation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode the observation from Unity environment results.

        Args:
            results: Raw results from Unity environment

        Returns:
            Processed observation dictionary with history (ordered from oldest to newest)
        """
        obs = results.obs
        # Process image observation (uint8 -> float32, normalized)
        image = np.array(obs[0][AGENT_ID] * 255, dtype=np.uint8)

        image = preprocess_image(image, mode=self.image_preprocess_mode)  # Returns CHW format

        # Extract state information
        state = obs[1][AGENT_ID]
        self.gripper_state = state[8]  # Update gripper state
        # state_with_rotmat = np.concatenate((state[4:7], quat_to_rotmat(state[0:4]).flatten()), axis=0)  # (3 + 9)
        trans, rotmat = self.get_robot_pose(results)

        # get translation of world frame in robot frame = -R^T @ t
        trans_world = -rotmat.T @ trans  # (3,)
        # state_with_rotmat = np.concatenate((trans, rotmat.flatten()), axis=0)  # (3 + 9)

        # Add current observation to history
        self.image_buffer.append(image)
        # self.state_buffer.append(state_with_rotmat)
        self.state_buffer.append(trans_world[:2])  # Only keep x and y (right-handed)

        # If history is not full, repeat the first observation
        while len(self.image_buffer) < self.history_length * self.obs_interval:
            self.image_buffer.append(image)
            # self.state_buffer.append(state_with_rotmat)
            self.state_buffer.append(trans_world[:2])  # Only keep x and y (right-handed)

        # Extract observations with the specified interval
        # We need to get the most recent observation and then go back by interval steps
        image_history = []
        state_history = []

        for i in range(self.history_length):
            # Calculate the index: start from the end, go back by i*interval steps
            idx = -1 - i * self.obs_interval
            # Use max to prevent index out of bounds errors
            idx = max(idx, -len(self.image_buffer))

            image_history.append(self.image_buffer[idx])
            state_history.append(self.state_buffer[idx])

        # Reverse the history to make it from oldest to newest
        image_history.reverse()
        state_history.reverse()

        # Stack history into arrays
        image_stack = np.stack(image_history, axis=0)
        state_stack = np.stack(state_history, axis=0)

        # Return observation dict with history
        if self.only_image:
            return {"image": image_stack, "state": np.zeros_like(state_stack)}
        elif self.only_state:
            return {"image": np.zeros_like(image_stack), "state": state_stack}
        return {"image": image_stack, "state": state_stack}

    def calc_z_axis_angle(self, robot_rotmat: np.ndarray) -> float:
        """
        Calculate the angle of the robot's z-axis against the world frame z-axis.
        This helps determine if the robot has flipped over.

        Args:
            robot_rotmat (np.ndarray): Rotation matrix of the robot

        Returns:
            float: Angle of the z-axis in radians (0 means upright, π means flipped)
        """
        # Get the z-axis vector from the rotation matrix (3rd column)
        robot_z_axis = robot_rotmat[:, 2]

        # World frame z-axis in right-handed coordinates (right, front, up)
        world_z_axis = np.array([0, 0, 1])

        # Calculate the dot product between the two vectors
        dot_product = np.dot(robot_z_axis, world_z_axis)

        # Clamp the dot product to avoid numerical errors
        dot_product = np.clip(dot_product, -1.0, 1.0)

        # Calculate the angle between the vectors in radians
        angle = np.arccos(dot_product)

        return angle

    def get_robot_pose(self, results: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract robot position and rotation from results.
        Note that original result is left-handed, we need to convert it to right-handed.
        Convert from left-handed (right, up, front) to right-handed (right, front, up).
        """
        obs = results.obs
        left_handed_position = obs[1][AGENT_ID][4:7]  # x, y, z for right, up and front

        # Convert position: [x, y, z] -> [x, z, y] (right, up, front -> right, front, up)
        right_handed_position = np.array(
            [
                left_handed_position[0],  # x (right) remains the same
                left_handed_position[2],  # z (front) becomes y (front)
                left_handed_position[1],  # y (up) becomes z (up)
            ]
        )

        # Get rotation matrix from quaternion
        left_handed_rotation = quat_to_rotmat(obs[1][AGENT_ID][0:4])

        # Convert rotation matrix for right-handed coordinate system
        # Swap columns and rows for y and z
        right_handed_rotation = np.array(
            [
                [left_handed_rotation[0, 0], left_handed_rotation[0, 2], left_handed_rotation[0, 1]],
                [left_handed_rotation[2, 0], left_handed_rotation[2, 2], left_handed_rotation[2, 1]],
                [left_handed_rotation[1, 0], left_handed_rotation[1, 2], left_handed_rotation[1, 1]],
            ]
        )

        return right_handed_position, right_handed_rotation

    def get_mineral_pose(self, results: Dict[str, Any]) -> np.ndarray:
        """Extract mineral position from results."""
        pose = results.obs[1][AGENT_ID][10:13]  # x, y, z for right, up and front
        # Convert position: [x, y, z] -> [x, z, y] (right, up, front -> right, front, up)
        right_handed_position = np.array(
            [
                pose[0],  # x (right) remains the same
                pose[2],  # z (front) becomes y (front)
                pose[1],  # y (up) becomes z (up)
            ]
        )
        return right_handed_position

    def calculate_distance_to_mineral(self, results: Dict[str, Any]) -> float:
        """Calculate distance between robot and mineral."""
        robot_pos = self.get_robot_pose(results)[0]
        mineral_pos = self.get_mineral_pose(results)
        return np.linalg.norm(robot_pos - mineral_pos)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Any:
        """Reset the environment to initial state."""
        if seed is not None:
            self.seed_value = seed
        if self.env is None:
            self.initialize_environment(self.seed_value)

        # Clear history
        self.image_buffer.clear()
        self.state_buffer.clear()
        z_angle = 180
        while abs(z_angle) > 10:
            self.env.reset()
            obs, _, _, info = self._step()
            z_angle = np.degrees(info["z_angle"])
        self.step_count = 0
        self.last_distance = self.calculate_distance_to_mineral(self.current_results)
        return obs

    def calculate_reward(self, results: Dict[str, Any]) -> float:
        """
        Calculate the reward based on current state and results.

        Includes both sparse reward from Unity and dense reward based on distance change.
        """
        sparse_reward = results.reward[AGENT_ID]
        if DEBUG and sparse_reward != 0:
            print(f"Sparse reward: {sparse_reward}")
        current_dist = self.calculate_distance_to_mineral(results)
        distance_reward = self.last_distance - current_dist
        self.last_distance = current_dist

        return sparse_reward + distance_reward

    def calculate_reward_enhanced(self, results: Dict[str, Any]) -> float:
        """
        Calculate reward based on https://arxiv.org/abs/1911.00357
        and https://arxiv.org/abs/2206.00997
        """
        sparse_reward = results.reward[AGENT_ID]
        # Assume positive sparse_reward for reach mineral
        success = 1 if sparse_reward > 0 else 0
        terminal_reward = 10 * success
        current_dist = self.calculate_distance_to_mineral(results)
        distance_reward = self.last_distance - current_dist

        # nearby = 1 if current_dist < 0.45 else 0
        # nearby_reward = 2.5 * (nearby - self.last_nearby)

        # enhanced_reward = terminal_reward + distance_reward + nearby_reward
        enhanced_reward = terminal_reward + distance_reward
        self.last_distance = current_dist
        # self.last_nearby = nearby
        return enhanced_reward

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, dict]:
        """
        Execute one environment step with given action.

        Args:
            action: Array of 3 values [vy, vx, vw] for robot control

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # debug: get action from keyboard: ws for action[0], ad for action[1], qe for action[2]
        if DEBUG:
            action[1], action[0], action[2] = get_action_from_keyboard()

        # Extend action with fixed arm angle and gripper control
        full_action = np.array([action[0], action[1], action[2], 10.0, 1.0], dtype=np.float32)

        action_tuple = ActionTuple(np.array([full_action]))
        obs, reward, done, info = self._step(action_tuple)

        # debug
        if DEBUG:
            trans, rotmat = self.get_robot_pose(self.current_results)
            z_angle = self.calc_z_axis_angle(rotmat)
            if self.current_results.reward[AGENT_ID] != 0:
                print(
                    f"Step: {self.step_count}, End Reward: {self.current_results.reward[AGENT_ID]}, done: {done}, "
                    f"Position: {(-rotmat.T @ trans)[:2]}, Abs Position: {trans[:2]}, Angle: {np.degrees(z_angle):.0f}, "
                )
            else:
                print(
                    f"Step: {self.step_count}, Reward: {reward}, done: {done}, "
                    f"Position: {(-rotmat.T @ trans)[:2]}, Abs Position: {trans[:2]}, Angle: {np.degrees(z_angle):.0f}, "
                )

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
        else:
            self.current_results = decision_steps
        obs = self.decode_observation(self.current_results)
        reward = self.calculate_reward(self.current_results)
        # reward = self.calculate_reward_enhanced(self.current_results)

        # Check if the robot is flipped over
        rotmat = self.get_robot_pose(self.current_results)[1]
        z_angle = self.calc_z_axis_angle(rotmat)
        info["z_angle"] = z_angle

        # Check episode length limit
        self.step_count += 1
        self.time_counter = time.perf_counter()
        if self.step_count >= self.max_episode_steps:
            done = True

        return obs, reward, done, info


def main():
    """Example usage of the environment."""
    env = EpMineEnv(port=3000, time_scale=1, render_size=(800, 600), max_episode_steps=32)
    obs = env.reset()
    done = False
    step = 0

    while True:
        while not done:
            action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            obs, reward, done, info = env.step(action)
            step += 1
        done = False
        obs = env.reset()


if __name__ == "__main__":
    main()
