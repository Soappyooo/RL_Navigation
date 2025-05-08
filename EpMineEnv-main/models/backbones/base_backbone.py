import torch
import torch.nn as nn
import numpy as np
import cv2


class VisualBackbone(nn.Module):
    def __init__(self, name: str, output_dim: int = 512):
        """
        Initialize the VisualBackbone class.

        Args:
            name (str): Name of the backbone architecture.
            output_dim (int): Output dimension of the backbone. Defaults to 512.
        """
        super(VisualBackbone, self).__init__()
        self.name = name
        self.model: nn.Module = None
        self.output_dim = output_dim
        self.output_projector: nn.Module = None

        if name == "efficientnet":
            from .efficientnet.model import EfficientNet

            self.model: EfficientNet = EfficientNet.from_name("efficientnet-b0")
            self.output_projector = nn.Linear(1280, output_dim)
        elif name == "convnext":
            pass  # TODO
        else:
            raise ValueError(f"Unsupported backbone architecture: {name}")

    def forward(self, x):
        """
        Forward pass through the backbone.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        if self.name == "efficientnet":
            x = self.model.extract_features(x)
            x = self.model._avg_pooling(x)
            x = x.view(x.size(0), -1)
            x = self.output_projector(x)
            return x
        else:
            raise ValueError(f"Unsupported backbone architecture: {self.name}")

    def load_backbone_state_dict(self, state_dict: dict):
        """
        Load the state dictionary into the backbone.

        Args:
            state_dict (dict): State dictionary to load.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        self.model.load_state_dict(state_dict)

    @staticmethod
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
        else:
            raise NotImplementedError(f"Preprocessing for mode '{mode}' is not implemented.")
