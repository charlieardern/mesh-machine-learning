from __future__ import annotations

import torch
from torch.utils.data import Dataset


class ModelNetSubset(Dataset):
    def __init__(
        self,
        train: bool,
        norm_params_x: tuple[torch.Tensor, torch.Tensor] | None,
        cats: list[int],
    ) -> None:
        self.cats = cats
        # Apply normalisation to x:
        if train:
            x = torch.load("data/ModelNet_subset/train_x.pt")
            y = torch.load("data/ModelNet_subset/train_y.pt")
        else:
            x = torch.load("data/ModelNet_subset/test_x.pt")
            y = torch.load("data/ModelNet_subset/test_y.pt")

        # Selecting the categories of interest:
        mask = torch.isin(y, torch.tensor(cats_of_interest)).squeeze()
        y = y[mask]
        x = x[mask]

        if norm_params_x is None:
            print(f"Shape with dim=1: {x.mean(dim=1).unsqueeze(1).shape}")
            print(f"Shape with dim=(0,1): {x.mean(dim=(0,1)).shape}")
            mean_x = x.mean(dim=1).unsqueeze(1)
            std_x = x.std(dim=(1,2)).reshape(-1,1,1)
            print(f"std_x.shape: {std_x.shape}")
            self.norm_params_x = mean_x, std_x
            self.x = (x - mean_x) / std_x
        else:
            self.norm_params_x = norm_params_x
            self.x = (x - norm_params_x[0]) / norm_params_x[1]

        # relabeling classes:
        for i in range(len(cats_of_interest)):
            mask = torch.isin(y, cats_of_interest[i])
            y[mask] = i
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_i = self.x[idx]
        y_i = self.y[idx]
        return x_i, y_i


# cats_of_interest = [0, 3, 5, 6, 7, 8, 9, 10, 15, 17, 19, 30, 35]
cats_of_interest = [0, 6, 8, 17, 35]

train_dataset = ModelNetSubset(train=True, norm_params_x=None, cats=cats_of_interest)
test_dataset = ModelNetSubset(
    train=False, norm_params_x=None, cats=cats_of_interest
)
