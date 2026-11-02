"""ResNet backbones with a new 37-class head, plus freezing helpers."""

import torch
import torch.nn as nn
from torchvision import models

from src.data import NUM_CLASSES

ARCHS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
}


def build_model(arch="resnet50", pretrained=True, num_classes=NUM_CLASSES):
    """Pretrained backbone, everything frozen, new trainable head."""
    builder, weights = ARCHS[arch]
    model = builder(weights=weights if pretrained else None)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)   # new layers are trainable
    return model


def unfreeze_layer4(model):
    for p in model.layer4.parameters():
        p.requires_grad = True


def set_train_mode(model):
    """train() for layer4 and the head; frozen parts keep their ImageNet BatchNorm stats."""
    model.train()
    for module in [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3]:
        module.eval()


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


def load_checkpoint(model, path):
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


def pick_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"
