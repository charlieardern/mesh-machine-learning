import torch
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# from dataset import dataset

# points = dataset[3].pos
points = torch.load("samples/object3_class_0.pt")

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
pyplot.savefig("test1")
