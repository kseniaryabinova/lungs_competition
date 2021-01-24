import os

import pandas as pd
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import ImageDataset, ValImageDataset
from resnet import ResNet18, ResNet34
from train_functions import one_batch_train, eval_model

torch.manual_seed(25)

df = pd.read_csv('train_with_split.csv')
train_df = df[df['split'] == 1]
train_image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(degrees=10),
])
train_set = ImageDataset(train_df, train_image_transforms, '../../mark/ranzcr/train', width_size=256)
train_loader = DataLoader(train_set, batch_size=6400, shuffle=True, num_workers=40, pin_memory=True)

val_df = df[df['split'] == 0]
val_image_transforms = transforms.Compose([transforms.ToTensor()])
val_set = ImageDataset(val_df, val_image_transforms, '../../mark/ranzcr/train', width_size=256)
val_loader = DataLoader(val_set, batch_size=6400, num_workers=40, pin_memory=True)

os.makedirs('checkpoints_34', exist_ok=True)

scaler = GradScaler()
if scaler is None:
    model = ResNet34(11, 1, pretrained_backbone=False, mixed_precision=False)
else:
    model = ResNet34(11, 1, pretrained_backbone=False, mixed_precision=True)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = [354.625, 23.73913043478261, 2.777105767812362, 110.32608695652173,
                 52.679245283018865, 9.152656621728786, 4.7851333032083145,
                 8.437891632878731, 2.4620064899945917, 0.4034751151063363, 31.534942820838626]
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weights).to(device))
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model = model.to(device)

for epoch in range(40):

    running_loss = 0.0
    model.train()

    for i, batch in enumerate(train_loader, 0):
        current_loss = one_batch_train(batch, model, optimizer, criterion, device, scaler)
        running_loss += current_loss

        if i % 1 == 0:
            print('[epoch %d, iteration %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
            running_loss = 0.0

    model.eval()
    total_loss, avg_auc,  = eval_model(model, val_loader, device, criterion)

    torch.save(model.state_dict(), 'checkpoints_34/model_epoch_{}_auc_{}_loss_{}.pth'.format(epoch, avg_auc, total_loss))
