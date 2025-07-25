from __future__ import annotations

import torch
from torch_geometric.datasets import ModelNet

print("Importing data...")
train_set = ModelNet(root="data/ModelNet", name="40", train=True)
test_set = ModelNet(root="data/ModelNet", name="40", train=False)
print("Complete")
print(len(train_set))
print(len(test_set))

new_size = 200  # New size of meshes
rejection_limit = 400  # Number of successive rejections before d gets halved

train_x = []
train_y = []

# Create training set:
for i in range(len(train_set)):
    data_y = train_set[i].y
    data_x_full = train_set[i].pos
    d = torch.tensor([1000.0])
    c = 0  # this counts number of successive rejections
    N = data_x_full.shape[0]
    samples = data_x_full[torch.randint(0, N, (1,))]  # create first data point

    while samples.shape[0] < new_size:
        idx = torch.randint(0, N, (1,))
        sample_i = data_x_full[idx]
        distances = torch.norm(samples - sample_i, dim=1)
        within_dist = (distances + 0.01 <= d).any()
        if not within_dist:
            samples = torch.cat((samples, sample_i), dim=0)
            c = 0
        else:
            c += 1
            if c > rejection_limit:
                c = 0
                d = d / 2
    print(f"Object {i + 1} saved with class {data_y.item()}")
    train_x.append(samples)
    train_y.append(data_y)
train_x = torch.stack(train_x, dim=0)
train_y = torch.stack(train_y, dim=0)

test_x = []
test_y = []

# Create testing set:
for i in range(len(test_set)):
    data_y = test_set[i].y
    data_x_full = test_set[i].pos
    d = torch.tensor([1000.0])
    c = 0  # this counts number of successive rejections
    N = data_x_full.shape[0]
    samples = data_x_full[torch.randint(0, N, (1,))]  # create first data point

    while samples.shape[0] < new_size:
        idx = torch.randint(0, N, (1,))
        sample_i = data_x_full[idx]
        distances = torch.norm(samples - sample_i, dim=1)
        within_dist = (distances + 0.01 <= d).any()
        if not within_dist:
            samples = torch.cat((samples, sample_i), dim=0)
            c = 0
        else:
            c += 1
            if c > rejection_limit:
                c = 0
                d = d / 2
    print(f"Object {i + 1} saved with class {data_y.item()}")
    test_x.append(samples)
    test_y.append(data_y)
test_x = torch.stack(test_x, dim=0)
test_y = torch.stack(test_y, dim=0)

torch.save(train_x, "data/ModelNet_subset/train_x.pt")
torch.save(train_y, "data/ModelNet_subset/train_y.pt")
torch.save(test_x, "data/ModelNet_subset/test_x.pt")
torch.save(test_y, "data/ModelNet_subset/test_y.pt")
print(f"train_x shape: {train_x.shape}")
print(f"train_y shape: {train_y.shape}")
print(f"test_x shape: {test_x.shape}")
print(f"test_y shape: {test_y.shape}")
