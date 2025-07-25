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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        return self.mlp(x)


class MLP2(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.map_to_latent = nn.Linear(in_features=3, out_features=latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
            # Returns the logits - softmax is evaluated outside the function
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"x in shape: {x.shape}")
        z = self.map_to_latent(x)
        # print(f"z1 shape: {z.shape}")
        z = torch.mean(z, dim=1)
        # print(f"z2 shape: {z.shape}")
        return self.mlp(z)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_heads,
        num_layers,
        hidden_dim,
        output_dim,
        max_points: int,
    ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim, output_dim)
        self.positional_encoding = nn.Parameter(
            torch.empty(1, max_points, hidden_dim).normal_()
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.3,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, num_points, _ = x.size()
        x = self.embedding(x)
        x = x + self.positional_encoding[:, :num_points, :]
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)
