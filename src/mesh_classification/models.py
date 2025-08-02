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
        input_dim: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        output_dim: int,
        max_points: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        # self.positional_encoding = nn.Parameter(
        #    torch.empty(1, max_points, hidden_dim).normal_()
        # )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.2,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, num_points, _ = x.size()
        x = self.embedding(x)
        # x = x + self.positional_encoding[:, :num_points, :]
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)


# Models from PCT Paper =================================================================================


class SPCTEmbedder(nn.Module):
    """
    This is my implementation of the embedding for the SPCT model from section 3.2, pg 4 in the paper:
    PCT: Point Cloud Transformer, M. Guo et. al. 2012.09688 https://arxiv.org/abs/2012.09688

    It is referred to as the naive point embedding in the paper.
    The paper uses embed_dim = 128

    Maps shape (B, N, 3) -> (B, N, embed_dim)
    """

    def __init__(self, embed_dim: int):
        self.embed_inputs = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(),
            nn.ReLU(),
        )

    def forward(self, points: torch.Tensor):
        return self.embed_inputs(points)


class OffsetAttention(nn.Module):
    """
    This is my implementation of the offset-attention mechanism from figure 3 in the paper:
    PCT: Point Cloud Transformer, M. Guo et. al. 2012.09688 https://arxiv.org/abs/2012.09688

    This is the 'Attention' block from figure 2, and features in both the SPCT and PCT models.
    The paper uses embed_dim = 128 and hidden_dim = 32

    Maps shape (B, N, embed_dim) -> (B, N, embed_dim)
    """

    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w_q = nn.Linear(embed_dim, hidden_dim)
        self.w_k = nn.Linear(embed_dim, hidden_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.lbr = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(), nn.ReLU()
        )

    def forward(self, f_in: torch.Tensor):
        q = self.w_q(f_in)  # (B, N, hidden_dim)
        k = self.w_k(f_in)  # (B, N, hidden_dim)
        v = self.w_v(f_in)  # (B, N, embed_dim)

        # Computing attention matrix
        a = k.transpose(-1, -2) @ q  # (B, N, N)
        a = nn.functional.softmax(
            a, dim=1
        )  # (B, N, N) apply softmax along dim=1 (approach used in paper - usually you do over dim=2)
        l1_norm = torch.sum(
            a, dim=2
        )  # (B, N, N) sum over dim=2 to get L1 norm of l columns
        attn_mat = (
            a / l1_norm
        )  # (B, N, N) ensures columns of attn_mat sum to 1 (we have to do this as the softmax is applied over the columns rather than rows like usual)

        f_sa = attn_mat @ v  # (B, N, embed_dim)

        return self.lbr(f_in - f_sa) + f_sa


class PCTEncoder(nn.Module):
    """
    This is my implementation of the encoder (minus the embedding) from figure 2 in the paper:
    PCT: Point Cloud Transformer, M. Guo et. al. 2012.09688 https://arxiv.org/abs/2012.09688

    It takes the embeded features as input and outputs the 'point feature' object. It can be used
    with both the SPCT and PCT embeddings.
    The paper uses num_layers = 4, embed_dim = 128 and hidden_dim = 32

    Maps shape (B, N, embed_dim) -> (B, N, num_layers * embed_dim)
    """

    def __init___(self, num_layers: int, embed_dim: int, hidden_dim: int):
        self.attn_layers = nn.ModuleList(
            [
                OffsetAttention(embed_dim=embed_dim, hidden_dim=hidden_dim)
                for _ in range(num_layers)
            ]
        )
        self.lbr_out = nn.Sequential(
            nn.Linear(num_layers * embed_dim, num_layers * embed_dim)
        )

    def forward(self, f_e: torch.Tensor):
        f_i = f_e  # (B, N, embed_dim)
        f_o = []
        for layer in self.attn_layers:
            f_i = layer(f_i)  # (B, N, embed_dim)
            f_o.append(f_i)
        f_o = torch.cat(f_o, dim=2)  # (B, N, num_layers * embed_dim)
        f_o = self.lbr_out(f_o)  # (B, N, num_layers * embed_dim)


class PCTClassifier(nn.Module):
    """
    This is my implementation of the classifier from figure 2 in the paper:
    PCT: Point Cloud Transformer, M. Guo et. al. 2012.09688 https://arxiv.org/abs/2012.09688

    It takes the 'point feature' as input and outputs the logits for the different categories
    The paper uses input_dim = 1024, hidden_dim = 256 and dropout = 0.5

    Maps shape (B, N, num_layers*embed_dim) -> (B, num_cats)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, num_cats):
        self.lbrd1 = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lbrd2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.linear_out = nn.Linear(hidden_dim, num_cats)

    def forward(self, f_in: torch.Tensor):
        max_pool, _ = torch.max(f_in, dim=1)  # (B, input_dim)
        av_pool, _ = torch.mean(f_in, dim=1)  # (B, input_dim)
        c = torch.cat([max_pool, av_pool], dim=1)  # (B, 2 * input_dim)
        c = self.lbrd1(f_in)  # (B, hidden_dim)
        c = self.lbrd2(c)  # (B, hidden_dim)
        return self.linear_out(c)  # (B, num_cats)


class SPCTClassifier(nn.Module):
    def __init__(self, embed_dim: int, attn_hidden_dim: int, num_attn_layers: int, ):
