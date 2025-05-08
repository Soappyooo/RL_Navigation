import torch
import torch.nn as nn
from .utils import PositionalEncoding
from typing import List


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
                nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True), num_layers=num_layers
            )
            self.positional_encoding = PositionalEncoding(dim, max_seq_len)
            self.cache_x: List[torch.Tensor] = []  # cache for inference
        elif name == "lstm":
            self.model = nn.LSTM(input_size=dim, num_layers=num_layers, hidden_size=dim, batch_first=True)
            self.cache_h: torch.Tensor = None  # cache for inference
            self.cache_c: torch.Tensor = None
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
        else:
            raise ValueError(f"Unsupported temporal encoder architecture: {self.name}")

    def infer(self, x, seq_len: int = 8, clear_cache: bool = False):
        """
        Inference method for the temporal encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).
            seq_len (int): Sequence length for Transformer. Defaults to 8.
            clear_cache (bool): Whether to clear the cache (e.g. start a new episode). Defaults to False.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, dim).
        """
        # check dimensions
        if len(x.shape) != 2:
            raise ValueError(f"Input tensor must be 2D, but got {len(x.shape)}D tensor.")
        if x.shape[1] != self.dim:
            raise ValueError(f"Input tensor must have last dimension of size {self.dim}, but got {x.shape[1]}.")

        if clear_cache:
            # clear the cache if specified
            if self.name == "transformer":
                self.cache_x = []
            elif self.name == "lstm":
                self.cache_h = None
                self.cache_c = None

        if self.name == "transformer":
            if len(self.cache_x) == 0:
                # if cache is empty, append the current input and repeat till seq_len
                self.cache_x = [x] * seq_len
            elif len(self.cache_x) == seq_len:
                # if cache is full, pop the first element and append the current input
                self.cache_x.pop(0)
                self.cache_x.append(x)
            else:
                # this should not happen
                raise ValueError(
                    f"Cache length {len(self.cache_x)} does not match the expected sequence length {seq_len}."
                )
            x = torch.stack(self.cache_x, dim=1)
            x = self.positional_encoding(x)
            x = self.model(x)
            return x[:, -1, :]
        elif self.name == "lstm":
            if self.cache_h is None:
                # if cache is empty, initialize the hidden and cell states
                self.cache_h = torch.zeros(self.num_layers, x.size(0), self.dim).to(x.device)
                self.cache_c = torch.zeros(self.num_layers, x.size(0), self.dim).to(x.device)
            x = x.unsqueeze(1)  # [batch_size, 1, dim] for LSTM
            x, (self.cache_h, self.cache_c) = self.model(x, (self.cache_h, self.cache_c))
            return x[:, -1, :]
        else:
            raise ValueError(f"Unsupported temporal encoder architecture: {self.name}")
