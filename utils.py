from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import Dataset

loss_fn = nn.CrossEntropyLoss()


def train_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    model.train()

    for x, y in data_loader:
        y_logits = model(x)
        # print(f"y_logits type: {y_logits.dtype}")
        # print(f"y type: {y.squeeze(1).dtype}")
        loss = loss_fn(y_logits, y.squeeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate_model(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_data: Dataset[tuple[torch.Tensor, torch.Tensor]],
    test_data: Dataset[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[float, float]:
    model.eval()
    with torch.inference_mode():
        train_loss = loss_fn(model(train_data.x), train_data.y.squeeze(1))
        test_loss = loss_fn(model(test_data.x), test_data.y.squeeze(1))
    print(f"Train loss: {train_loss:.5f}")
    print(f"Test loss: {test_loss:.5f}")
    return train_loss, test_loss


def unnormalise_data(
    x: torch.Tensor,
    x_params: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    x = x_params[1] * x + x_params[0]
    return x
