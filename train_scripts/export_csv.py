import torch
from torch.utils.data import DataLoader

from dataloader import ImageDataset
from resnet18 import ResNet18


model = ResNet18(11, 1, pretrained_backbone=False, mixed_precision=False)
model.load_state_dict(torch.load('model.pth'))
model.eval()

