import torch
import torch.nn as nn
from .utils import PositionalEncoding
from typing import List, Union
import torch.nn.functional as F
import numpy as np


class TemporalEncoder(nn.Module):
    def __init__(self, name: str, dim: int = 512, max_seq_len: int = 8, num_layers: int = 2):
        """
        Initialize the TemporalEncoder class.

        Args:
            name (str): Name of the temporal encoder architecture.
            dim (int): Input, hidden and output dimension of the encoder. Defaults to 512.
            max_seq_len (int): Maximum sequence length for positional encoding (Transformer only). Defaults to 8.
            num_layers (int): Number of layers in the encoder. Defaults to 2.
        """
        super(TemporalEncoder, self).__init__()
        self.name = name
        self.model: nn.Module = None
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers

        if name == "transformer":
            self.model = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=dim, nhead=4, batch_first=True, dim_feedforward=dim * 4, activation=F.silu
                ),
                num_layers=num_layers,
            )
            self.positional_encoding = PositionalEncoding(dim, max_seq_len)
            self.cache_x: List[torch.Tensor] = []  # cache for inference
        elif name == "lstm":
            self.model = nn.LSTM(input_size=dim, num_layers=num_layers, hidden_size=dim, batch_first=True)
            self.cache_h: torch.Tensor = None  # cache for inference
            self.cache_c: torch.Tensor = None
        elif name == "identity":
            self.model = nn.Identity()

        else:
            raise ValueError(f"Unsupported temporal encoder architecture: {name}")

    def forward(self, x):
        """
        Forward pass through the temporal encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, dim) (the last token).
        """
        # check dimensions
        if len(x.shape) != 3:
            raise ValueError(f"Input tensor must be 3D, but got {len(x.shape)}D tensor.")
        if x.shape[2] != self.dim:
            raise ValueError(f"Input tensor must have last dimension of size {self.dim}, but got {x.shape[2]}.")

        if self.name == "transformer":
            x = self.positional_encoding(x)
            x = self.model(x)
            return x[:, -1, :]
        elif self.name == "lstm":
            x, _ = self.model(x)
            return x[:, -1, :]
        elif self.name == "identity":
            assert x.shape[1] == 1, "Identity encoder only supports seq_len=1"
            return x.squeeze(1)  # [batch_size, 1, dim] -> [batch_size, dim] *only support seq_len=1*
        else:
            raise ValueError(f"Unsupported temporal encoder architecture: {self.name}")

    def infer(self, x: torch.Tensor, seq_len: int = 8, clear_cache: Union[bool, np.ndarray] = False):
        """
        Inference method for the temporal encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).
            seq_len (int): Sequence length for Transformer. Defaults to 8.
            clear_cache (Union[bool, np.ndarray]): Whether to clear the cache (e.g. start a new episode). Defaults to False.
                When an array is provided, it should have the same length as the batch size.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, dim).
        """
        # check dimensions
        if x.dim() != 2:
            raise ValueError(f"Input tensor must be 2D, but got {x.dim()}D tensor.")
        if x.shape[1] != self.dim:
            raise ValueError(f"Input tensor must have last dimension of size {self.dim}, but got {x.shape[1]}.")

        batch_size = x.shape[0]

        # make clear_cache an array if it's a boolean
        if isinstance(clear_cache, bool):
            clear_cache = np.full(batch_size, clear_cache, dtype=bool)
        # check if clear_cache is an array
        if not isinstance(clear_cache, np.ndarray):
            raise ValueError(f"clear_cache must be a boolean or a numpy array, but got {type(clear_cache)}.")
        # check if clear_cache is a 1D array
        if len(clear_cache) != batch_size:
            raise ValueError(
                f"When providing an array for clear_cache, its length ({len(clear_cache)}) "
                f"must match the batch size ({batch_size})."
            )
        # if batch_size != cache batch_size, initialize cache
        if self.name == "transformer" and len(self.cache_x) != batch_size:
            self.cache_x = []
        elif self.name == "lstm" and (self.cache_h is not None and self.cache_h.shape[1] != batch_size):
            self.cache_h = None
            self.cache_c = None

        if self.name == "transformer":
            # TODO: check if this works
            if len(self.cache_x) == 0:
                # Initialize empty cache with repeated current input using deep copy
                self.cache_x = [x.clone() for _ in range(seq_len)]

                for i, should_clear in enumerate(clear_cache):
                    if should_clear:
                        for t in range(len(self.cache_x)):
                            self.cache_x[t][i] = x[i]

            # Update cache in FIFO manner
            self.cache_x.pop(0)
            self.cache_x.append(x.clone())

            # Process with transformer
            x = torch.stack(self.cache_x, dim=1)
            x = self.positional_encoding(x)
            x = self.model(x)
            return x[:, -1, :]

        elif self.name == "lstm":
            if self.cache_h is None:
                # Initialize empty cache
                self.cache_h = torch.zeros(self.num_layers, batch_size, self.dim).to(x.device)
                self.cache_c = torch.zeros(self.num_layers, batch_size, self.dim).to(x.device)
            else:
                # Clear hidden and cell states for specific samples
                for i, should_clear in enumerate(clear_cache):
                    if should_clear:
                        self.cache_h[:, i, :] = 0
                        self.cache_c[:, i, :] = 0

            # Process with LSTM
            x = x.unsqueeze(1)  # [batch_size, 1, dim] for LSTM
            x, (self.cache_h, self.cache_c) = self.model(x, (self.cache_h, self.cache_c))
            return x[:, -1, :]  # [batch_size, 1, dim] -> [batch_size, dim]

        else:
            raise ValueError(f"Unsupported temporal encoder architecture: {self.name}")
