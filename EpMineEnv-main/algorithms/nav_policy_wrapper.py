import torch as th
import torch.nn as nn
import numpy as np
import gym
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path

from models.nav_policy import NavPolicy


class NavActorCriticPolicy(ActorCriticPolicy):
    """
    Custom policy for navigation task that wraps NavPolicy and adds pose prediction.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: callable,
        backbone_name: str = "efficientnet",
        encoder_name: str = "transformer",
        hidden_dim: int = 512,
        head_hidden_dims: Tuple[int] = (256, 128),
        max_seq_len: int = 8,
        from_pretrained: bool = False,
        pretrained_backbone_path: Optional[Path] = None,
        **kwargs,
    ):
        """
        Initialize the NavActorCriticPolicy.
        Args:
            observation_space: The observation space of the environment.
            action_space: The action space of the environment.
            lr_schedule: Learning rate schedule.
            backbone_name: Name of the visual backbone architecture.
            encoder_name: Name of the temporal encoder architecture.
            hidden_dim: Hidden dimension of the policy.
            head_hidden_dims: Tuple of hidden dimensions for the heads.
            max_seq_len: Maximum sequence length for positional encoding.
            from_pretrained: Whether to load a pretrained backbone.
            pretrained_backbone_path: Path to the pretrained backbone weights.
            **kwargs: Additional arguments for the parent class.
        """
        self.nav_policy: NavPolicy = None
        self.max_seq_len = max_seq_len
        self.backbone_name = backbone_name
        self.encoder_name = encoder_name
        self.hidden_dim = hidden_dim
        self.head_hidden_dims = head_hidden_dims
        self.action_dim = action_space.shape[0]
        self.from_pretrained = from_pretrained
        self.pretrained_backbone_path = pretrained_backbone_path

        # _build() will be called in the parent class
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to parent
            **kwargs,
        )

        if self.from_pretrained:
            if self.pretrained_backbone_path is None:
                raise ValueError("pretrained_backbone_path must be provided when from_pretrained is True.")
            # Load the pretrained backbone
            self.nav_policy.visual_backbone.load_backbone_state_dict(
                th.load(self.pretrained_backbone_path, map_location=self.device, weights_only=True)
            )
            print(f"Loaded pretrained backbone from {self.pretrained_backbone_path.resolve()}")

    def forward(self, obs: Dict[str, th.Tensor], deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor, critic and pose)

        Args:
            obs: Dictionary containing:
                - image: tensor of shape (batch_size, seq_len, channels, height, width)
                - state: tensor of shape (batch_size, seq_len, 12) or None
            deterministic: Whether to sample or use deterministic actions

        Returns:
            actions, values, log_probs
        """
        actions, values, _ = self.nav_policy(obs, output_pose=False)
        distribution = self._get_action_dist_from_latent(actions)

        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()

        log_probs = distribution.log_prob(actions).sum(dim=-1)  # Sum log_probs to get joint probability

        return actions, values, log_probs

    def evaluate_actions(
        self, obs: Dict[str, th.Tensor], actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        Args:
            obs: Dictionary containing:
                - image: tensor of shape (batch_size, seq_len, channels, height, width)
                - state: tensor of shape (batch_size, seq_len, 12) or None
            actions: Tensor of shape (batch_size, n_actions)

        Returns:
            values, log_probs, entropy, pose_pred
        """
        latent_pi, values, pose_pred = self.nav_policy(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)

        # For Diagonal Gaussian, we need to sum the log_probs across action dimensions
        log_prob = distribution.log_prob(actions).sum(dim=-1)  # Sum log_probs to get joint probability
        entropy = distribution.entropy().sum(dim=-1)  # Sum entropy across action dimensions

        return values, log_prob, entropy, pose_pred

    def predict_values(self, obs):
        """
        Predict the value given the observation.

        Args:
            obs: Dictionary containing:
                - image: tensor of shape (batch_size, seq_len, channels, height, width)
                - state: tensor of shape (batch_size, seq_len, 12) or None

        Returns:
            values: Tensor of shape (batch_size,)
        """
        _, values, _ = self.nav_policy(obs, output_action=False, output_pose=False)
        return values

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> th.distributions.Distribution:
        """
        Retrieve action distribution given the latent features.
        For continuous actions, we use a Diagonal Gaussian distribution with learned std.
        Note that the log_prob from this distribution will be the sum of log_probs for each dimension.
        """
        mean_actions = latent_pi

        # Create Gaussian distribution with learned std
        log_std = self.log_std
        std = th.ones_like(mean_actions) * th.exp(log_std)
        return th.distributions.Normal(mean_actions, std)

    def _build(self, lr_schedule) -> None:
        """
        Create the model and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.nav_policy = NavPolicy(
            backbone_name=self.backbone_name,
            encoder_name=self.encoder_name,
            hidden_dim=self.hidden_dim,
            head_hidden_dims=self.head_hidden_dims,
            max_seq_len=self.max_seq_len,
        ).to(self.device)

        # log_std for the action distribution
        self.log_std = nn.Parameter(th.ones(self.action_dim) * 0, requires_grad=True)

        # Create the optimizer
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def extract_features(self, obs: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        """
        Prepare observation for the neural network.
        The image has already been preprocessed in the environment.
        """
        return obs

    def predict(
        self,
        observation: Dict[str, th.Tensor],
        clear_cache: bool = False,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Get the policy action from an observation.

        Args:
            obs: Dictionary containing:
                - image: tensor of shape (channels, height, width)
                - state: tensor of shape (12) or None
            clear_cache: Whether to clear the cache (e.g. start a new episode).
            deterministic: Whether or not to return deterministic actions.

        Returns:
            the model's action
        """
        with th.no_grad():
            # Convert observation to tensor
            obs = th.as_tensor(observation["image"]).float().to(self.device)
            # add batch dimension if needed
            if obs.dim() == 3:
                obs = obs.unsqueeze(0)
            # check if obs is 4D
            if obs.dim() != 4:
                if obs.dim() == 5:  # (batch_size, seq_len, channels, height, width)
                    obs = obs[0, -1, :, :, :].unsqueeze(0)  # take the last frame
                else:
                    raise ValueError(f"Image tensor must be 4D, but got {obs.dim()}D tensor.")
            # check if obs is 3 channels
            if obs.shape[1] != 3:
                raise ValueError(f"Image tensor must have channels=3, but got {obs.shape[1]}.")

            actions = self.nav_policy.predict(
                {"image": obs, "state": None},
                seq_len=self.max_seq_len,
                clear_cache=clear_cache,
            )
            distribution = self._get_action_dist_from_latent(actions)

            if deterministic:
                actions = distribution.mean
            else:
                actions = distribution.sample()

        return actions.numpy(force=True)  # [1, action_dim]
