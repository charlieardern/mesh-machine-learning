from __future__ import annotations

import torch
from torch import nn


class BasicMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
            # Returns the logits - softmax is evaluated outside the function
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(x)
