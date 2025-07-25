from __future__ import annotations

import torch
from matplotlib import pyplot

from dataset import test_dataset, train_dataset
from utils import unnormalise_data

x_normed = train_dataset.x
print(f"x_normed element (0,0): {x_normed[:,0,0]}")

x = unnormalise_data(train_dataset.x, train_dataset.norm_params_x)
print(f"x element (0,0): {x[:,0,0]}")
# points = dataset[3].pos
item_index = 1
points = x[item_index]
print(f"category:{train_dataset.y[item_index]}")

x, y, z = points[:, 0], points[:, 1], points[:, 2]

fig = pyplot.figure()
ax = fig.add_subplot(111, projection="3d")
ax.view_init(elev=90, azim=90)
ax.scatter(x, y, z, c="blue", s=1)

ax.set_xlabel("X")
ax.set_xlabel("Y")
ax.set_xlabel("Z0")

ax.set_box_aspect([1, 1, 1])
scale = torch.tensor([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
ax.set_xlim([x.mean() - scale / 2, x.mean() + scale / 2])
ax.set_ylim([y.mean() - scale / 2, y.mean() + scale / 2])
ax.set_zlim([z.mean() - scale / 2, z.mean() + scale / 2])
pyplot.savefig("figures/test1")
