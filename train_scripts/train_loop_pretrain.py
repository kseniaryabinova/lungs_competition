import os
import shutil
import time
from pytz import timezone
from datetime import datetime

from efficient_net_sa import EfficientNetSA
from vit import ViT

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':64:8'

import pandas as pd
import numpy as np

import torch

from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from albumentations.pytorch import ToTensorV2
import albumentations as alb

from adas_optimizer import Adas
from dataloader import ImageDataset, ChestXDataset
from resnet import ResNet18, ResNet34
from efficient_net import EfficientNet
from train_functions import one_epoch_train, eval_model, group_weight

torch.manual_seed(25)
np.random.seed(25)

os.makedirs('tensorboard_runs', exist_ok=True)
shutil.rmtree('tensorboard_runs')
writer = SummaryWriter(log_dir='tensorboard_runs', filename_suffix=str(time.time()))

width_size = 600

df = pd.read_csv('data.csv')
train_df = df[df['fold'] != 1]
train_image_transforms = alb.Compose([
    alb.HorizontalFlip(p=0.5),
    # alb.CLAHE(p=0.5),
    # alb.OneOf([
    #     alb.GridDistortion(
    #         num_steps=8,
    #         distort_limit=0.5,
    #         p=1.0
    #     ),
    #     alb.OpticalDistortion(
    #         distort_limit=0.5,
    #         shift_limit=0.5,
    #         p=1.0,
    #     ),
    #     alb.ElasticTransform(alpha=3, p=1.0)],
    #     p=0.5
    # ),
    alb.RandomResizedCrop(
        height=width_size,
        width=width_size,
        scale=(0.8, 1.2),
        p=0.5
    ),
    alb.ShiftScaleRotate(shift_limit=0.025, scale_limit=0.1, rotate_limit=20, p=0.5),
    # alb.HueSaturationValue(
    #     hue_shift_limit=20,
    #     sat_shift_limit=20,
    #     val_shift_limit=20,
    #     p=0.5
    # ),
    # alb.RandomBrightnessContrast(
    #     brightness_limit=(-0.15, 0.15),
    #     contrast_limit=(-0.15, 0.15),
    #     p=0.5
    # ),
    # alb.CoarseDropout(
    #     max_holes=12,
    #     min_holes=6,
    #     max_height=int(width_size / 6),
    #     max_width=int(width_size / 6),
    #     min_height=int(width_size / 20),
    #     min_width=int(width_size / 20),
    #     p=0.5
    # ),
    alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
train_set = ChestXDataset(train_df, train_image_transforms, '../data', width_size=width_size)
data = train_set[0]
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=48, pin_memory=True, drop_last=True)

val_df = df[df['fold'] == 1]
val_image_transforms = alb.Compose([
    alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
val_set = ChestXDataset(val_df, val_image_transforms, '../data', width_size=width_size)
val_loader = DataLoader(val_set, batch_size=16, num_workers=48, pin_memory=True, drop_last=True)

checkpoints_dir_name = 'tf_efficientnet_b7_ns_pretrain_{}'.format(width_size)
os.makedirs(checkpoints_dir_name, exist_ok=True)

# model = ResNet18(11, 1, pretrained_backbone=True, mixed_precision=True)
model = EfficientNet(15, pretrained_backbone=True, mixed_precision=True, model_name='tf_efficientnet_b7_ns')
# model = ViT(11, pretrained_backbone=True, mixed_precision=True, model_name='vit_base_patch16_384')
# model = EfficientNetSA(11, pretrained_backbone=True, mixed_precision=True, model_name='tf_efficientnet_b5_ns')

scaler = GradScaler()
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# class_weights = [354.625, 23.73913043478261, 2.777105767812362, 110.32608695652173,
#                  52.679245283018865, 9.152656621728786, 4.7851333032083145,
#                  8.437891632878731, 2.4620064899945917, 0.4034751151063363, 31.534942820838626]
# class_names = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
#                'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
#                'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present']

# criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weights).to(device))
criterion = torch.nn.BCEWithLogitsLoss()
# criterion = BCEwithLabelSmoothing(pos_weights=torch.tensor(class_weights).to(device))
# optimizer = Adas(model.parameters())
optimizer = Adam(group_weight(model, weight_decay=1e-4), lr=1e-4, weight_decay=0)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6, last_epoch=-1)
model = model.to(device)

for epoch in range(50):
    total_train_loss, train_avg_auc, train_auc, train_data_pr, train_duration = one_epoch_train(
        model, train_loader, optimizer, criterion, device, scaler, iters_to_accumulate=8, clip_grads=False)
    total_val_loss, val_avg_auc, val_auc, val_data_pr, val_duration = eval_model(
        model, val_loader, device, criterion, scaler)
    scheduler.step()

    writer.add_scalars('avg/loss', {'train': total_train_loss, 'val': total_val_loss}, epoch)
    writer.add_scalars('avg/auc', {'train': train_avg_auc, 'val': val_avg_auc}, epoch)
    writer.flush()

    print('EPOCH %d:\tTRAIN [duration %.3f sec, loss: %.3f, avg auc: %.3f]\t\t'
          'VAL [duration %.3f sec, loss: %.3f, avg auc: %.3f]\tCurrent time %s' %
          (epoch + 1, train_duration, total_train_loss, train_avg_auc,
           val_duration, total_val_loss, val_avg_auc, str(datetime.now(timezone('Europe/Moscow')))))

    torch.save(model.state_dict(),
               os.path.join(checkpoints_dir_name, '{}_epoch{}_val_auc{}_loss{}_train_auc{}_loss{}.pth'.format(
                   checkpoints_dir_name, epoch + 1, round(val_avg_auc, 3), round(total_val_loss, 3),
                   round(train_avg_auc, 3), round(total_train_loss, 3))))

writer.close()
