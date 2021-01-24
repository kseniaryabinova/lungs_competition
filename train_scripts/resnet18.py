import torch.nn as nn
from torch.cuda.amp import autocast

from torchvision import models


class ResNet18(nn.Module):
    def __init__(self, n_classes, num_input_channel, pretrained_backbone, mixed_precision):
        super().__init__()
        self.amp = mixed_precision
        self.classifier = models.resnet18(pretrained=pretrained_backbone)
        self.classifier.conv1 = nn.Conv2d(num_input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.classifier.fc = nn.Linear(512, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.amp:
            with autocast():
                x = self.classifier(x)
        else:
            x = self.classifier(x)
        return x

    def inference(self, x):
        if self.amp:
            with autocast():
                x = self.sigmoid(self.classifier(x))
        else:
            x = self.sigmoid(self.classifier(x))
        return x


class ResNet34(nn.Module):
    def __init__(self, n_classes, num_input_channel, pretrained_backbone, mixed_precision):
        super().__init__()
        self.amp = mixed_precision
        self.classifier = models.resnet34(pretrained=pretrained_backbone)
        self.classifier.conv1 = nn.Conv2d(num_input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.classifier.fc = nn.Linear(512, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.amp:
            with autocast():
                x = self.classifier(x)
        else:
            x = self.classifier(x)
        return x

    def inference(self, x):
        if self.amp:
            with autocast():
                x = self.sigmoid(self.classifier(x))
        else:
            x = self.sigmoid(self.classifier(x))
        return x
