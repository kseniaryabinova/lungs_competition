import os
import shutil
import time
from pytz import timezone
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':64:8'
os.environ['WANDB_SILENT'] = 'true'

import pandas as pd
import numpy as np

import torch

from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from albumentations.pytorch import ToTensorV2
import albumentations as alb

import wandb

from dataloader import ImageDataset, ChestXDataset, NoisyStudentDataset
from efficient_net import EfficientNet, EfficientNetNoisyStudent
from train_functions import one_epoch_train, eval_model, group_weight

torch.manual_seed(25)
np.random.seed(25)

os.makedirs('tensorboard_runs', exist_ok=True)
shutil.rmtree('tensorboard_runs')
writer = SummaryWriter(log_dir='tensorboard_runs', filename_suffix=str(time.time()))
wandb.init(project='effnet7', group=wandb.util.generate_id())

width_size = 640
wandb.config.width_size = width_size
wandb.config.aspect_rate = 1

batch_size = 12
accumulation_step = 10
wandb.config.batch_size = batch_size
wandb.config.accumulation_step = accumulation_step

ranzcr_df = pd.read_csv('train_folds.csv')
ranzcr_train_df = ranzcr_df[ranzcr_df['fold'] != 1]

chestx_df = pd.read_csv('chestx_pseudolabeled_data_lazy_balancing.csv')
train_image_transforms = alb.Compose([
    alb.ImageCompression(quality_lower=65, p=0.5),
    alb.HorizontalFlip(p=0.5),
    alb.CLAHE(p=0.5),
    alb.OneOf([
        alb.GridDistortion(
            num_steps=8,
            distort_limit=0.5,
            p=1.0
        ),
        alb.OpticalDistortion(
            distort_limit=0.5,
            shift_limit=0.5,
            p=1.0,
        ),
        alb.ElasticTransform(alpha=3, p=1.0)],
        p=0.7
    ),
    alb.RandomResizedCrop(
        height=width_size,
        width=width_size,
        scale=(0.8, 1.2),
        p=0.7
    ),
    alb.RGBShift(p=0.5),
    alb.RandomSunFlare(p=0.5),
    alb.RandomFog(p=0.5),
    alb.RandomBrightnessContrast(p=0.5),
    alb.HueSaturationValue(
        hue_shift_limit=20,
        sat_shift_limit=20,
        val_shift_limit=20,
        p=0.5
    ),
    alb.ShiftScaleRotate(shift_limit=0.025, scale_limit=0.1, rotate_limit=20, p=0.5),
    alb.CoarseDropout(
        max_holes=12,
        min_holes=6,
        max_height=int(width_size / 6),
        max_width=int(width_size / 6),
        min_height=int(width_size / 6),
        min_width=int(width_size / 20),
        p=0.5
    ),
    alb.IAAAdditiveGaussianNoise(loc=0, scale=(2.5500000000000003, 12.75), per_channel=False, p=0.5),
    alb.IAAAffine(scale=1.0, translate_percent=None, translate_px=None, rotate=0.0, shear=0.0, order=1, cval=0,
                  mode='reflect', p=0.5),
    alb.IAAAffine(rotate=90., p=0.5),
    alb.IAAAffine(rotate=180., p=0.5),
    alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
wandb.config.train_augmentations = str(train_image_transforms.transforms.transforms)
train_set = NoisyStudentDataset(ranzcr_train_df, chestx_df, train_image_transforms,
                                '../ranzcr/train', '../data', width_size=width_size)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=48, pin_memory=True, drop_last=True)

ranzcr_valid_df = ranzcr_df[ranzcr_df['fold'] == 1]
valid_image_transforms = alb.Compose([
    alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
valid_set = ImageDataset(ranzcr_valid_df, valid_image_transforms, '../ranzcr/train', width_size=width_size)
valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=48, pin_memory=True, drop_last=False)

checkpoints_dir_name = 'tf_efficientnet_b7_ns_noisy_student_{}'.format(width_size)
os.makedirs(checkpoints_dir_name, exist_ok=True)
wandb.config.model_name = checkpoints_dir_name

model = EfficientNetNoisyStudent(11, pretrained_backbone=True, mixed_precision=True, model_name='tf_efficientnet_b7_ns')

scaler = GradScaler()
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.config.is_loss_weights = 'no'
class_names = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
               'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
               'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present']

criterion = torch.nn.BCEWithLogitsLoss()

lr_start = 3e-4
lr_end = 1e-6
weight_decay = 0
epoch_num = 40
wandb.config.lr_start = lr_start
wandb.config.lr_end = lr_end
wandb.config.weight_decay = weight_decay
wandb.config.epoch_num = epoch_num
wandb.config.optimizer = 'adam'
wandb.config.scheduler = 'CosineAnnealingLR'

optimizer = Adam(model.parameters(), lr=lr_start, weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=lr_end, last_epoch=-1)
model = model.to(device)

max_val_auc = 0

for epoch in range(epoch_num):
    train_loss, train_avg_auc, train_auc, train_rocs, train_data_pr, train_duration = one_epoch_train(
        model, train_loader, optimizer, criterion, device, scaler,
        iters_to_accumulate=accumulation_step, clip_grads=False)
    val_loss, val_avg_auc, val_auc, val_rocs, val_data_pr, val_duration = eval_model(
        model, valid_loader, device, criterion, scaler)
    scheduler.step()

    wandb.log({'train_loss': train_loss, 'val_loss': val_loss,
               'train_auc': train_avg_auc, 'val_auc': val_avg_auc, 'epoch': epoch})
    for class_name, auc1, auc2 in zip(class_names, train_auc, val_auc):
        wandb.log({'{} train auc'.format(class_name): auc1,
                   '{} val auc'.format(class_name): auc2, 'epoch': epoch})

    if val_avg_auc > max_val_auc:
        max_val_auc = val_avg_auc
        wandb.run.summary["best_accuracy"] = val_avg_auc

    print('EPOCH %d:\tTRAIN [duration %.3f sec, loss: %.3f, avg auc: %.3f]\t\t'
          'VAL [duration %.3f sec, loss: %.3f, avg auc: %.3f]\tCurrent time %s' %
          (epoch + 1, train_duration, train_loss, train_avg_auc,
           val_duration, val_loss, val_avg_auc, str(datetime.now(timezone('Europe/Moscow')))))

    torch.save(model.state_dict(),
               os.path.join(checkpoints_dir_name, '{}_epoch{}_val_auc{}_loss{}_train_auc{}_loss{}.pth'.format(
                   checkpoints_dir_name, epoch + 1, round(val_avg_auc, 3), round(val_loss, 3),
                   round(train_avg_auc, 3), round(train_loss, 3))))

writer.close()
