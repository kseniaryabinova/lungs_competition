import torch.nn as nn
from torch.cuda.amp import autocast

from torchvision import models


class ResNet18(nn.Module):
    def __init__(self, n_classes, pretrained_backbone, mixed_precision):
        super().__init__()
        self.amp = mixed_precision
        self.classifier = models.resnet18(pretrained=pretrained_backbone)
        self.classifier.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        if self.amp:
            with autocast():
                x = self.classifier(x)
        else:
            x = self.classifier(x)
        return x
