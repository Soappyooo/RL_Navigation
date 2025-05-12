import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConv(nn.Module):
    """
    Simple convolutional feature extractor. Assumes input size of (3, 128, 128).
    """

    def __init__(self):
        """
        Initialize the ConvFeatureExtractor class.
        This class implements a convolutional feature extractor.
        """
        super(SimpleConv, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        x = F.silu(self.conv3(x))  # (batch_size, 128, 12, 12)
        x = F.adaptive_avg_pool2d(x, (4, 4))  # (batch_size, 128, 4, 4)
        return x
