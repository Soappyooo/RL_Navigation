import torch
import torch.nn as nn
from .temporal_encoders import TemporalEncoder
from .backbones import VisualBackbone
from typing import Dict, Optional, List, Union, Literal
import numpy as np


def rot9d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """
    Symmetric orthogonalization and build rotation matrix from a 9D vector
    Args:
        x: 9D vector of shape (N, 9)
    Returns:
        Rotation matrix of shape (N, 3, 3)
    """
    assert x.dim() <= 2, "Input tensor must be 1D or 2D"
    assert x.size(-1) == 9, "Last dimension must be of size 9"
    m = x.view(-1, 3, 3)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    r = torch.matmul(u, vt)
    return r


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple = (512,),
        activation: Literal["SiLU", "ReLU", "Tanh"] = "SiLU",
    ):
        """
        Initialize the MLP class.
        This class implements a multi-layer perceptron with SiLU activations.

        Args:
            input_dim (int): Input dimension of the MLP.
            output_dim (int): Output dimension of the MLP.
            hidden_dims (tuple): Tuple of hidden dimensions. Defaults to (512,).
        """
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        activations = {
            "SiLU": nn.SiLU(),
            "ReLU": nn.ReLU(),
            "Tanh": nn.Tanh(),
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation function: {activation}. Supported: {list(activations.keys())}")
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activations[activation])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def uniform_init(self, final_layer_scale: float = 0.01):
        """
        Initialize the weights of the MLP using uniform distribution.
        Args:
            final_layer_scale (float): Scale for the weight of final layer. Defaults to 0.01.
        """
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                stdv = 1.0 / (layer.weight.size(1) ** 0.5)
                if layer == self.model[-1]:
                    nn.init.uniform_(layer.weight, -stdv * final_layer_scale, stdv * final_layer_scale)
                else:
                    nn.init.uniform_(layer.weight, -stdv, stdv)
                nn.init.zeros_(layer.bias)


class NavPolicy(nn.Module):
    def __init__(
        self,
        backbone_name: Literal["efficientnet", "simple"],
        encoder_name: Literal["identity", "lstm", "transformer", "mlp"],
        hidden_dim: int = 512,
        head_hidden_dims: tuple = (256, 128),
        seq_len: int = 1,
        pose_auxiliary_mode: Literal["naive", "concatenate"] = "naive",
    ):
        """
        Initialize the Navigation Policy class.
        This class combines a visual backbone, a temporal encoder and several heads.
        Heads include an actor head (output action), a critic head (output value) and pose estimation head.

        Args:
            backbone_name (str): Name of the visual backbone architecture.
            encoder_name (str): Name of the temporal encoder architecture.
            hidden_dim (int): Hidden dimension of the policy. Defaults to 512.
            head_hidden_dims (tuple): Tuple of hidden dimensions for the heads. Defaults to (256, 128).
            max_seq_len (int): Sequence length for image and state inputs. Defaults to 1.
            pose_auxiliary_mode (str): Mode for pose auxiliary task. Defaults to "naive".
        """
        super(NavPolicy, self).__init__()
        self.pose_auxiliary_mode = pose_auxiliary_mode

        self.visual_backbone = VisualBackbone(backbone_name, hidden_dim)
        self.temporal_encoder = TemporalEncoder(encoder_name, hidden_dim, seq_len)

        if pose_auxiliary_mode == "naive":
            self.actor_head = MLP(hidden_dim, 3, head_hidden_dims, activation="Tanh")
            # Use uniform initialization and make last layer small according to https://arxiv.org/pdf/2006.05990
            self.actor_head.uniform_init(final_layer_scale=0.01)
            self.critic_head = MLP(hidden_dim, 1, head_hidden_dims, activation="SiLU")
            self.pose_head = MLP(hidden_dim, 2, head_hidden_dims, activation="SiLU")

        elif pose_auxiliary_mode == "concatenate":
            pose_feat_dim = 32
            self.actor_head = MLP(hidden_dim + pose_feat_dim, 3, head_hidden_dims, activation="Tanh")
            # Use uniform initialization and make last layer small according to https://arxiv.org/pdf/2006.05990
            self.actor_head.uniform_init(final_layer_scale=0.01)
            self.critic_head = MLP(hidden_dim + pose_feat_dim, 1, head_hidden_dims, activation="SiLU")
            # separate pose visual backbone
            self.pose_visual_backbone = VisualBackbone(backbone_name, hidden_dim)
            self.pose_head = MLP(hidden_dim * seq_len, 2, head_hidden_dims, activation="SiLU")
            self.pose_projection = nn.Linear(2, pose_feat_dim)  # project pose to hidden_dim
            # concatenate pose projection and output from temporal encoder
            # self.concatenate_layer = nn.Sequential(nn.Linear(hidden_dim + 32, hidden_dim))

        else:
            raise ValueError(f"Unsupported pose auxiliary mode: {pose_auxiliary_mode}")

        self.hidden_dim = hidden_dim
        self.head_hidden_dims = head_hidden_dims
        self.seq_len = seq_len
        self.backbone_name = backbone_name
        self.encoder_name = encoder_name
        self.action_low = torch.tensor([-10.0, -10.0, -3.0])
        self.action_high = torch.tensor([10.0, 10.0, 3.0])
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_mean = (self.action_high + self.action_low) / 2.0

    def _unnormalize_action(self, normalized_action: torch.Tensor) -> torch.Tensor:
        """
        WE HAVE MOVED THIS FUNCTION TO nav_policy_wrapper.py
        Convert normalized action (from tanh, in [-1,1]) to actual action space
        """
        if normalized_action.device != self.action_scale.device:
            self.action_scale = self.action_scale.to(normalized_action.device)
            self.action_mean = self.action_mean.to(normalized_action.device)

        return self.action_mean + normalized_action * self.action_scale

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        output_value: bool = True,
        output_pose: bool = True,
        output_action: bool = True,
    ) -> tuple:
        """
        Forward pass through the navigation policy.
        Args:
            x (Dict[str, torch.Tensor]): Input dictionary containing:
                - image: tensor of shape (batch_size, seq_len, channels, height, width)
                - state: tensor of shape (batch_size, seq_len, 2) or None
            output_value (bool): If True, compute the value (critic head). Defaults to True.
            output_pose (bool): If True, compute the pose (pose head). Defaults to True.
            output_action (bool): If True, compute the action (actor head). Defaults to True.
        Returns:
            tuple: Tuple of action, value and pose tensors.
        """
        # Process image input
        image = x["image"]
        if len(image.shape) != 5:
            raise ValueError(f"Image tensor must be 5D, but got {len(image.shape)}D tensor.")
        if image.shape[2] != 3:
            raise ValueError(f"Image tensor must have channels=3, but got {image.shape[2]}.")

        batch_size = image.size(0)
        seq_len = image.size(1)

        # reshape to (batch_size * seq_len, channels, height, width)
        image = image.view(batch_size * seq_len, *image.shape[2:])

        # pass through visual backbone
        feat = self.visual_backbone(image)  # (batch_size * seq_len, hidden_dim)

        # reshape to (batch_size, seq_len, hidden_dim)
        feat = feat.view(batch_size, seq_len, -1)

        # pass through temporal encoder
        feat = self.temporal_encoder(feat)  # (batch_size, hidden_dim)

        if self.pose_auxiliary_mode == "naive":
            # pass through critic head
            if output_value:
                value = self.critic_head(feat)  # (batch_size, 1)
            else:
                value = None

            # pass through actor head
            if output_action:
                action = self.actor_head(feat)  # (batch_size, 3)
                action = torch.tanh(action)  # normalize to [-1,1]
            else:
                action = None

            # pass through pose head
            if output_pose:
                pose = self.pose_head(feat)  # (batch_size, 3 + 9)
            else:
                pose = None

        elif self.pose_auxiliary_mode == "concatenate":
            # get pose
            pose_feat = self.pose_visual_backbone(image)  # (batch_size * seq_len, hidden_dim)
            pose_feat = pose_feat.view(batch_size, -1)  # (batch_size, hidden_dim * seq_len)
            pose = self.pose_head(pose_feat)  # (batch_size, 2)

            # pose_projection = self.pose_projection(pose.detach())  # (batch_size, hidden_dim), detach to avoid gradient
            pose_projection = self.pose_projection(x["state"][:, -1, :].float() / 3)  # use real pose and normalize
            feat = torch.cat((pose_projection, feat), dim=1)  # (batch_size, hidden_dim - 32 + 32)
            # pass through critic head
            if output_value:
                value = self.critic_head(feat)
            else:
                value = None
            # pass through actor head
            if output_action:
                action = self.actor_head(feat)
                action = torch.tanh(action)
            else:
                action = None

        return action, value, pose

    def forward_rollout_deprecated(
        self,
        x: Dict[str, torch.Tensor],
        output_value: bool = True,
        output_pose: bool = True,
        output_action: bool = True,
        dones: Optional[np.ndarray] = None,
    ) -> tuple:
        """
        [Deprecated]
        Forward pass through the navigation policy. This is used for rollout,
        which means that observations are consecutive, so we use the newest observation
        and cache the previous ones.

        Args:
            x (Dict[str, torch.Tensor]): Input dictionary containing:
                - image: tensor of shape (batch_size, seq_len, channels, height, width)
                - state: tensor of shape (batch_size, seq_len, 12) or None
            output_value (bool): If True, compute the value (critic head). Defaults to True.
            output_pose (bool): If True, compute the pose (pose head). Defaults to True.
            output_action (bool): If True, compute the action (actor head). Defaults to True.
            dones (Optional[np.ndarray]): Array of booleans indicating whether each sequence is done. Defaults to None.
        Returns:
            tuple: Tuple of actor, critic and pose tensors.
        """
        # Process image input
        image = x["image"]
        if len(image.shape) != 5:
            raise ValueError(f"Image tensor must be 5D, but got {len(image.shape)}D tensor.")
        if image.shape[2] != 3:
            raise ValueError(f"Image tensor must have channels=3, but got {image.shape[2]}.")

        batch_size = image.size(0)
        seq_len = image.size(1)

        # extract newest (batch_size, channels, height, width)
        image = image[:, -1, :, :, :]

        # pass through visual backbone
        feat = self.visual_backbone(image)  # (batch_size, hidden_dim)

        # pass through temporal encoder
        feat = self.temporal_encoder.infer(feat, seq_len, clear_cache=dones)

        # pass through critic head
        if output_value:
            value = self.critic_head(feat)  # (batch_size, 1)
        else:
            value = None

        # pass through actor head
        if output_action:
            action = self.actor_head(feat)  # (batch_size, 3)
            action = torch.tanh(action)  # normalize to [-1,1]
            # we scale it after normal distribution, moving line below to nav_policy_wrapper.py
            # action = self._unnormalize_action(action)  # scale to actual action space
        else:
            action = None

        # pass through pose head
        if output_pose:
            pose = self.pose_head(feat)  # (batch_size, 3 + 9)

            # svd not implemented for AMP (half precision)
            # pose[:, 3:] = rot9d_to_rotmat(pose[:, 3:]).view(-1, 9)  # reconstruct rotation matrix
        else:
            pose = None

        return action, value, pose

    def predict(
        self, x: Dict[str, torch.Tensor], seq_len: int = 8, clear_cache: Union[bool, np.ndarray] = False
    ) -> torch.Tensor:
        """
        Predict the action and value given the input.
        Args:
            x (Dict[str, torch.Tensor]): Input dictionary containing:
                - image: tensor of shape (batch_size, channels, height, width)
                - state: tensor of shape (batch_size, 12) or None
            seq_len (int): Sequence length for Transformer. Defaults to 8.
            clear_cache (bool | array): Whether to clear the cache (e.g. start a new episode). Defaults to False.
        Returns:
            torch.Tensor: Predicted action tensor of shape (batch_size, 3).
        """
        image = x["image"]
        if len(image.shape) != 4:
            raise ValueError(f"Image tensor must be 4D, but got {len(image.shape)}D tensor.")
        if image.shape[1] != 3:
            raise ValueError(f"Image tensor must have channels=3, but got {image.shape[1]}.")

        batch_size = image.size(0)
        feat = self.visual_backbone(image)  # (batch_size, hidden_dim)
        feat = self.temporal_encoder.infer(feat, seq_len, clear_cache)  # (batch_size, hidden_dim)
        action = self.actor_head(feat)
        action = torch.tanh(action)  # normalize to [-1,1]
        # we scale it after normal distribution, moving line below to nav_policy_wrapper.py
        # action = self._unnormalize_action(action)  # scale to actual action space
        return action
