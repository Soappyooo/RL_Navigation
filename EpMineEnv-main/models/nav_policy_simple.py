import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple = (512,),
        activation: str = "SiLU",
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
