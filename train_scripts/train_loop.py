import datetime
import os
import shutil
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import pandas as pd
import numpy as np

import torch

from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from albumentations.pytorch import ToTensorV2, ToTensor
import albumentations as alb

from adas_optimizer import Adas
from dataloader import ImageDataset
from resnet import ResNet18, ResNet34
from train_functions import one_epoch_train, eval_model

torch.manual_seed(25)
np.random.seed(25)

shutil.rmtree('tensorboard_runs')
writer = SummaryWriter(log_dir='tensorboard_runs', filename_suffix=str(time.time()))

df = pd.read_csv('train_with_split.csv')
train_df = df[df['split'] == 1]
train_image_transforms = alb.Compose([
    # alb.CLAHE(p=0.5),
    # alb.GridDistortion(p=0.5),
    alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
train_set = ImageDataset(train_df, train_image_transforms, '../ranzcr/train', width_size=128)
# train_set = ImageDataset(train_df, train_image_transforms, '../dataset/train', width_size=128)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=48, pin_memory=True)

val_df = df[df['split'] == 0]
val_image_transforms = alb.Compose([
    alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
val_set = ImageDataset(val_df, val_image_transforms, '../ranzcr/train', width_size=128)
# val_set = ImageDataset(val_df, val_image_transforms, '../dataset/train', width_size=128)
val_loader = DataLoader(val_set, batch_size=64, num_workers=48, pin_memory=True)

os.makedirs('checkpoints', exist_ok=True)

scaler = GradScaler()
# scaler = None
model = ResNet18(11, 1, pretrained_backbone=True, mixed_precision=True)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = [354.625, 23.73913043478261, 2.777105767812362, 110.32608695652173,
                 52.679245283018865, 9.152656621728786, 4.7851333032083145,
                 8.437891632878731, 2.4620064899945917, 0.4034751151063363, 31.534942820838626]
class_names = [
    'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
    'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
    'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present'
]
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weights).to(device))
optimizer = Adas(model.parameters())
model = model.to(device)

for epoch in range(40):
    total_train_loss, train_avg_auc, train_auc, train_duration = one_epoch_train(
        model, train_loader, optimizer, criterion, device, scaler)

    total_val_loss, val_avg_auc, val_auc, val_duration = eval_model(
        model, val_loader, device, criterion, scaler)

    writer.add_scalars('avg/loss', {'train': total_train_loss, 'val': total_val_loss}, epoch)
    writer.add_scalars('avg/auc', {'train': train_avg_auc, 'val': val_avg_auc}, epoch)
    for class_name, auc1, auc2 in zip(class_names, train_auc, val_auc):
        writer.add_scalars('AUC/{}'.format(class_name), {'train': auc1, 'val': auc2}, epoch)

    print('EPOCH %d:\tTRAIN [duration %.3f sec, loss: %.3f, avg auc: %.3f]\t\t'
          'VAL [duration %.3f sec, loss: %.3f, avg auc: %.3f]\tCurrent time %s' %
          (epoch + 1, train_duration, total_train_loss, train_avg_auc,
           val_duration, total_val_loss, val_avg_auc, str(datetime.datetime.now())))
    print('{}\n{}'.format(str(train_auc), str(val_auc)))

    torch.save(model.state_dict(), 'checkpoints/model_epoch_{}_auc_{}_loss_{}.pth'.format(
        epoch + 1, round(val_avg_auc, 2), round(total_val_loss, 2)))

writer.close()
