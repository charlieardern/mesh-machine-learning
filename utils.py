from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

loss_fn = nn.CrossEntropyLoss()


def train_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    model.to(device)
    model.train()

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
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
    model.to(device)
    model.eval()
    train_data_x = train_data.x.to(device)
    train_data_y = train_data.y.to(device)
    test_data_x = test_data.x.to(device)
    test_data_y = test_data.y.to(device)

    with torch.inference_mode():
        train_loss = loss_fn(model(train_data_x), train_data_y.squeeze(1))
        test_loss = loss_fn(model(test_data_x), test_data_y.squeeze(1))
    print(f"Train loss: {train_loss:.5f}")
    print(f"Test loss: {test_loss:.5f}")
    return train_loss, test_loss


def unnormalise_data(
    x: torch.Tensor,
    x_params: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    x = x_params[1] * x + x_params[0]
    return x
