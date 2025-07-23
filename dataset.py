from __future__ import annotations

import torch
from torch.utils.data import Dataset


class ModelNetSubset(Dataset):
    def __init__(
        self,
        train: bool,
        norm_params_y: tuple[torch.Tensor, torch.Tensor] | None,
        norm_params_x: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> None:
        # Apply normalisation to x:
        if train:
            x = torch.load("data/ModelNet_subset/train_x.pt")
            y = torch.load("data/ModelNet_subset/train_y.pt")
        else:
            x = torch.load("data/ModelNet_subset/test_x.pt")
            y = torch.load("data/ModelNet_subset/test_y.pt")

        if norm_params_x is None:
            mean_x = x.mean(dim=0)
            std_x = x.std(dim=0)
            self.norm_params_x = mean_x, std_x
            self.x = (x - mean_x) / std_x
        else:
            self.norm_params_x = norm_params_x
            self.x = (x - norm_params_x[0]) / norm_params_x[1]

        # Apply normalisation to y:
        if norm_params_y is None:
            mean_y = y.mean(dim=0)
            std_y = y.std(dim=0)
            self.norm_params_y = mean_y, std_y
            self.y = (y - mean_y) / std_y
        else:
            self.norm_params_y = norm_params_y
            self.y = (y - norm_params_y[0]) / norm_params_y[1]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_i = self.x[idx]
        y_i = self.y[idx]
        return x_i, y_i


train_dataset = ModelNetSubset(train=True, norm_params_x=None, norm_params_y=None)
test_dataset = ModelNetSubset(
    train=False,
    norm_params_x=train_dataset.norm_params_x,
    norm_params_y=train_dataset.norm_params_y,
)
