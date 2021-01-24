import os

import pandas as pd
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import ImageDataset
from export_csv import ValImageDataset
from resnet18 import ResNet18
from train_functions import one_batch_train, eval_model

torch.manual_seed(25)

train_df = pd.read_csv('../dataset/train.csv')
image_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
])
train_set = ImageDataset(train_df, image_transforms, '../dataset/train')
train_loader = DataLoader(train_set, batch_size=16, num_workers=12, pin_memory=True)

image_transforms = transforms.Compose([transforms.ToTensor()])
val_set = ValImageDataset(image_transforms, '../dataset/test')
val_loader = DataLoader(val_set, batch_size=32, num_workers=12, pin_memory=True)

os.makedirs('checkpoints', exist_ok=True)

scaler = GradScaler()
if scaler is None:
    model = ResNet18(11, 1, pretrained_backbone=False, mixed_precision=False)
else:
    model = ResNet18(11, 1, pretrained_backbone=False, mixed_precision=True)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_weights = [354.625, 23.73913043478261, 2.777105767812362, 110.32608695652173,
                 52.679245283018865, 9.152656621728786, 4.7851333032083145,
                 8.437891632878731, 2.4620064899945917, 0.4034751151063363, 31.534942820838626]
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weights).to(device))
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model = model.to(device)

for epoch in range(1):

    running_loss = 0.0
    model.train()

    for i, batch in enumerate(train_loader, 0):
        current_loss = one_batch_train(batch, model, optimizer, criterion, device, scaler)
        running_loss += current_loss

        if i % 100 == 0:
            print('[epoch %d, iteration %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
            running_loss = 0.0

    model.eval()
    avg_auc = eval_model(model, train_loader, device, criterion)

    torch.save(model.state_dict(), 'checkpoints/model_epoch_{}_auc_{}.pth'.format(epoch, avg_auc))
