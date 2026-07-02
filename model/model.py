# model.py
# ResNet18 classifier architecture.

import torch.nn as nn
from torchvision import models


def build_resnet34(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Build the ResNet34 classifier with a checkpoint/training-driven head size.
    """
    weights = models.ResNet34_Weights.DEFAULT if pretrained else None
    model = models.resnet34(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    # Replace the ImageNet 1000-class head with the training/checkpoint class count.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model