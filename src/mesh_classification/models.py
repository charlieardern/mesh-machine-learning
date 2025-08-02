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
        z = self.map_to_latent(x)
        z = torch.mean(z, dim=1)
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
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)


# Models from PCT Paper =================================================================================
class BatchNormLastDim(nn.Module):
    """
    Regular BatchNorm1d which now works on shape (B, N, D) rather than (B, D, N)
    """
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
    def forward(self, x):
        x = x.permute(0,2,1) # (B, D, N)
        x = self.bn(x) # (B, D, N)
        return x.permute(0,2,1) # (B, N, D)


class SPCTEmbedder(nn.Module):
    """
    This is my implementation of the embedding for the SPCT model from section 3.2, pg 4 in the paper:
    PCT: Point Cloud Transformer, M. Guo et. al. 2012.09688 https://arxiv.org/abs/2012.09688

    It is referred to as the naive point embedding in the paper.
    The paper uses embed_dim = 128

    Maps shape (B, N, 3) -> (B, N, embed_dim)
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_inputs = nn.Sequential(
            nn.Linear(3, embed_dim),
            BatchNormLastDim(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            BatchNormLastDim(embed_dim),
            nn.ReLU(),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
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
            nn.Linear(embed_dim, embed_dim), BatchNormLastDim(embed_dim), nn.ReLU()
        )

    def forward(self, f_in: torch.Tensor) -> torch.Tensor:
        q = self.w_q(f_in)  # (B, N, hidden_dim)
        k = self.w_k(f_in)  # (B, N, hidden_dim)
        v = self.w_v(f_in)  # (B, N, embed_dim)

        # Computing attention matrix
        a = q @ k.transpose(-1, -2)  # (B, N, N)
        a = nn.functional.softmax(
            a, dim=1
        )  # (B, N, N) apply softmax along dim=1 (approach used in paper - usually you do over dim=2)
        l1_norm = torch.sum(
            a, dim=2
        ).unsqueeze(2)  # (B, N, N) sum over dim=2 to get L1 norm of l columns
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

    def __init__(self, num_layers: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.attn_layers = nn.ModuleList(
            [
                OffsetAttention(embed_dim=embed_dim, hidden_dim=hidden_dim)
                for _ in range(num_layers)
            ]
        )
        self.lbr_out = nn.Sequential(
            nn.Linear(num_layers * embed_dim, num_layers * embed_dim)
        )

    def forward(self, f_e: torch.Tensor) -> torch.Tensor:
        f_i = f_e  # (B, N, embed_dim)
        f_o = []
        for layer in self.attn_layers:
            f_i = layer(f_i)  # (B, N, embed_dim)
            f_o.append(f_i)
        f_o = torch.cat(f_o, dim=2)  # (B, N, num_layers * embed_dim)
        return self.lbr_out(f_o)  # (B, N, num_layers * embed_dim)


class PCTClassifier(nn.Module):
    """
    This is my implementation of the classifier from figure 2 in the paper:
    PCT: Point Cloud Transformer, M. Guo et. al. 2012.09688 https://arxiv.org/abs/2012.09688

    It takes the 'point feature' as input and outputs the logits for the different classes
    The paper uses input_dim = 1024, hidden_dim = 256 and dropout = 0.5

    Maps shape (B, N, num_layers*embed_dim) -> (B, num_classes)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, num_classes) -> None:
        super().__init__()
        self.lbrd1 = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            BatchNormLastDim(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lbrd2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            BatchNormLastDim(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.linear_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, f_in: torch.Tensor) -> torch.Tensor:
        max_pool, _ = torch.max(f_in, dim=1)  # (B, input_dim)
        av_pool = torch.mean(f_in, dim=1)  # (B, input_dim)
        c = torch.cat([max_pool, av_pool], dim=1)  # (B, 2 * input_dim)
        c = c.unsqueeze(1) # (B, 1, hidden_dim)
        c = self.lbrd1(c)  # (B, 1, hidden_dim)
        c = self.lbrd2(c)  # (B, 1, hidden_dim)
        return self.linear_out(c).squeeze(1) # (B, num_classes)


class SPCTMeshClassifier(nn.Module):
    """
    This is my implementation of the full SPCT classifier from the paper:
    PCT: Point Cloud Transformer, M. Guo et. al. 2012.09688 https://arxiv.org/abs/2012.09688

    It takes the point cloud as input and outputs the logits for each different class,
    to be passed through a softmax

    Maps shape (B, N, 3) -> (B, num_classes)
    """
    def __init__(self, embed_dim: int, attn_hidden_dim: int, num_attn_layers: int, classifier_hidden_dim: int, classifier_dropout: float, num_classes: int) -> None:
        super().__init__()
        self.embedding = SPCTEmbedder(embed_dim=embed_dim)
        self.encoder = PCTEncoder(num_layers=num_attn_layers,embed_dim=embed_dim, hidden_dim=attn_hidden_dim)
        self.classifier = PCTClassifier(input_dim=4*embed_dim, hidden_dim=classifier_hidden_dim, dropout=classifier_dropout, num_classes=num_classes)

    def forward(self, mesh) -> torch.Tensor:
        x = self.embedding(mesh) # (B, N, embed_dim)
        x = self.encoder(x) # (B, N, num_layers*embed_dim)
        return self.classifier(x) # (B, num_classes)