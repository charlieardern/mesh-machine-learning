from __future__ import annotations

from timeit import default_timer

import pathlib
import numpy
import torch
from matplotlib import pyplot
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import test_dataset, train_dataset
from models import TransformerClassifier
from utils import evaluate_model, loss_fn, train_step


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    #model = MLP2(input_dim=600, latent_dim=64, hidden_dim=128, output_dim=len(train_dataset.cats))
    model = TransformerClassifier(
        input_dim=3,
        num_heads=4,
        num_layers=6,
        hidden_dim=256,
        output_dim=len(train_dataset.cats),
        max_points=200,
    )

    #optimizer = torch.optim.Adam(params=model.parameters(), lr=0.1)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    def print_train_time(
        start: float, end: float, device: torch.device = None
    ) -> float:
        total_time = end - start
        print(f"Train time on {device}: {total_time:.3f} seconds")
        return total_time

    train_time_start = default_timer()
    epochs = 60
    train_losses = []
    test_losses = []
    test_accuracies = []
    best_test_loss = 9e9

    weights_directory = pathlib.Path("saved_objects")
    weights_directory.mkdir(exist_ok=True)

    figures_directory = pathlib.Path("figures")
    figures_directory.mkdir(exist_ok=True)

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch + 1}\n----------")
        train_step(
            model=model,
            data_loader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        scheduler.step()
        train_loss, test_loss, accuracy = evaluate_model(
            model=model,
            loss_fn=loss_fn,
            train_data=train_dataset,
            test_data=test_dataset,
        )
        train_losses.append(train_loss.detach().item())
        test_losses.append(test_loss.detach().item())
        test_accuracies.append(accuracy)
        if (epoch > epochs / 4) and (test_loss < best_test_loss):
            best_test_loss = test_loss
            torch.save(model.state_dict(), (weights_directory / "model_1_weights.pth"))
            print(f"New best: {best_test_loss}")
    train_time_end = default_timer()
    print_train_time(start=train_time_start, end=train_time_end, device=device)

    pyplot.plot(numpy.arange(0, epochs, step=1), train_losses, label="train loss")
    pyplot.plot(numpy.arange(0, epochs, step=1), test_losses, label="test loss")
    pyplot.ylabel("loss")
    pyplot.xlabel("epochs")
    pyplot.legend()
    pyplot.savefig(figures_directory / "loss_curves")
    print(f"Weights saved with lowest cross entropy test loss of {best_test_loss}")


if __name__ == "__main__":
    main()
