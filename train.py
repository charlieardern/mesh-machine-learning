from __future__ import annotations

from timeit import default_timer

import numpy
import torch
from matplotlib import pyplot
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import test_dataset, train_dataset
from models import BasicMLP
from utils import evaluate_model, loss_fn, train_step


def main() -> None:
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    model = BasicMLP(input_dim=600, hidden_dim=64, output_dim=len(train_dataset.cats))

    optimizer = torch.optim.Adam(params=model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.66)

    def print_train_time(start: float, end: float) -> float:
        total_time = end - start
        print(f"Train time: {total_time:.3f} seconds")
        return total_time

    train_time_start = default_timer()
    epochs = 50
    train_losses = []
    test_losses = []
    # best_test_loss = 9e9

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch + 1}\n----------")
        train_step(
            model=model,
            data_loader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        scheduler.step()
        train_loss, test_loss = evaluate_model(
            model=model,
            loss_fn=loss_fn,
            train_data=train_dataset,
            test_data=test_dataset,
        )
        train_losses.append(train_loss.detach().item())
        test_losses.append(test_loss.detach().item())
    train_time_end = default_timer()
    print_train_time(start=train_time_start, end=train_time_end)

    pyplot.plot(numpy.arange(0, epochs, step=1), train_losses, label="train loss")
    pyplot.plot(numpy.arange(0, epochs, step=1), test_losses, label="test loss")
    pyplot.ylabel("loss")
    pyplot.xlabel("epochs")
    pyplot.legend()
    pyplot.savefig("figures/loss_curves")


if __name__ == "__main__":
    main()
