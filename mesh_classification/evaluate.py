from __future__ import annotations

import torch
from matplotlib import pyplot
from torch_geometric.datasets import ModelNet

from mesh_classification.dataset import test_dataset, train_dataset
from mesh_classification.utils import unnormalise_data
from mesh_classification.models import TransformerClassifier

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

#plot full size mesh:
train_set = ModelNet(root="data/ModelNet", name="40", train=True)
points = train_set[4].pos

# Plot object in 3D
x = unnormalise_data(train_dataset.x, train_dataset.norm_params_x)
item_index = 1
#points = x[item_index]
print(f"category:{train_dataset.y[item_index]}")

x, y, z = points[:, 0], points[:, 1], points[:, 2]

fig = pyplot.figure(figsize=(20,20))
ax = fig.add_subplot(111, projection="3d")
ax.set_axis_off()
ax.view_init(elev=10, azim=155)
ax.scatter(x, y, z, c="black", s=0.1, alpha=1.0)

ax.set_xlabel("X")
ax.set_xlabel("Y")
ax.set_xlabel("Z")

ax.set_box_aspect([1, 1, 1])
scale = torch.tensor([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
ax.set_xlim([x.mean() - scale / 4, x.mean() + scale / 4])
ax.set_ylim([y.mean() - scale / 4, y.mean() + scale / 4])
ax.set_zlim([z.mean() - scale / 4, z.mean() + scale / 4])
fig.tight_layout()
pyplot.savefig("figures/test2")

model = TransformerClassifier(input_dim=3,num_heads=4,num_layers=6,hidden_dim=256,output_dim=len(train_dataset.cats),max_points=200,)
model.load_state_dict(torch.load("saved_objects/model_1_weights.pth"))
model.eval()

test_data_x = test_dataset.x
test_data_y = test_dataset.y

with torch.inference_mode():
        y_pred = torch.softmax(model(test_data_x), dim=1).argmax(dim=1)

class_names = ["airplane", "bench", "bottle", "bowl", "car", "chair", "cone", "cup", "flower_pot", "guitar", "lamp", "sofa", "sink"]

print(f"y_pred shape: {y_pred.shape}")
print(f"y shape: {test_data_y.reshape(-1).shape}")

print(f"y_pred: {y_pred}")
print(f"y: {test_data_y.reshape(-1)}")

confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_pred, target=torch.tensor(test_data_y.reshape(-1)))


fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(), class_names=class_names, figsize=(10,7))
pyplot.savefig("figures/confusion_matrix")


