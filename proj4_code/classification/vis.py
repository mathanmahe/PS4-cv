import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import ImageFolder

from proj4_code.classification.dl_utils import predict_labels


def visualize(model: torch.nn.Module, split: str, data_transforms, data_base_path: str = "../data") -> None:
    loader = ImageFolder(os.path.join(data_base_path, split), transform=data_transforms)
    loader_ = ImageFolder(os.path.join(data_base_path, split))

    class_labels = loader.class_to_idx
    class_labels = {ele.lower(): class_labels[ele] for ele in class_labels}
    labels = {class_labels[ele]: ele for ele in class_labels}
    selected_indices = random.choices(range(len(loader)), k=4)
    fig, axs = plt.subplots(2, 2)
    for i in range(4):
        img_tensor, gt_label = loader.__getitem__(selected_indices[i])
        img, _ = loader_.__getitem__(selected_indices[i])
        with torch.no_grad():
            output = model(img_tensor.unsqueeze(0).to(next(model.parameters()).device))
            predicted = predict_labels(output).item()
        axs[i // 2, i % 2].imshow(img, cmap="gray")
        axs[i // 2, i % 2].set_title("Predicted:{}|Correct:{}".format(labels[predicted], labels[gt_label]))
        axs[i // 2, i % 2].axis("off")
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.5)
    plt.show()
