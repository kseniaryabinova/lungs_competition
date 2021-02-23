import os
import shutil
import time
from pytz import timezone
from datetime import datetime

from efficient_net_sa import EfficientNetSA
from vit import ViT

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':64:8'
os.environ['WANDB_SILENT'] = 'true'

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

import wandb

from adas_optimizer import Adas
from dataloader import ImageDataset, UnlabeledImageDataset
from resnet import ResNet18, ResNet34
from efficient_net import EfficientNet
from train_functions import one_epoch_train, eval_model, group_weight, UnlabeledLoss, train_one_epoch_pseudolabel

torch.manual_seed(25)
np.random.seed(25)

os.makedirs('tensorboard_runs', exist_ok=True)
shutil.rmtree('tensorboard_runs')
writer = SummaryWriter(log_dir='tensorboard_runs', filename_suffix=str(time.time()))
wandb.init(project='effnet5', group=wandb.util.generate_id())

width_size = 512
wandb.config.width_size = width_size
wandb.config.aspect_rate = 1

batch_size = 16
accumulation_step = 1
wandb.config.batch_size = batch_size
wandb.config.accumulation_step = accumulation_step

df = pd.read_csv('train_folds.csv')
train_df = df[df['fold'] != 1]
annot_df = pd.read_csv('../ranzcr/train_annotations.csv')
train_image_transforms = alb.Compose([
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
        p=0.5
    ),
    alb.RandomResizedCrop(
        height=width_size,
        width=width_size,
        scale=(0.5, 1.5),
        p=0.5
    ),
    alb.ShiftScaleRotate(shift_limit=0.025, scale_limit=0.1, rotate_limit=20, p=0.5),
    alb.HueSaturationValue(
        hue_shift_limit=20,
        sat_shift_limit=20,
        val_shift_limit=20,
        p=0.5
    ),
    alb.RandomBrightnessContrast(
        brightness_limit=(-0.15, 0.15),
        contrast_limit=(-0.15, 0.15),
        p=0.5
    ),
    alb.CoarseDropout(
        max_holes=12,
        min_holes=6,
        max_height=int(width_size / 6),
        max_width=int(width_size / 6),
        min_height=int(width_size / 20),
        min_width=int(width_size / 20),
        p=0.5
    ),
    alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
wandb.config.train_augmentations = str(train_image_transforms.transforms.transforms)
train_set = ImageDataset(train_df, train_image_transforms, '../../mark/ranzcr/train', width_size=width_size)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=48, pin_memory=True, drop_last=True)

val_df = df[df['fold'] == 1]
val_image_transforms = alb.Compose([
    alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
val_set = ImageDataset(val_df, val_image_transforms, '../../mark/ranzcr/train', width_size=width_size)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=48, pin_memory=True, drop_last=True)

unlabeled_set = UnlabeledImageDataset(val_image_transforms, '../ranzcr/test', width_size=width_size)
unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, num_workers=48,
                                  pin_memory=True, drop_last=True)
wandb.config.is_pseudolabel = 'yes'

checkpoints_dir_name = 'tf_efficientnet_b5_ns_pseudolabel'.format(width_size)
os.makedirs(checkpoints_dir_name, exist_ok=True)
wandb.config.model_name = checkpoints_dir_name

model = EfficientNet(11, pretrained_backbone=True, mixed_precision=True, model_name='tf_efficientnet_b5_ns')

scaler = GradScaler()
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# class_weights = [354.625, 23.73913043478261, 2.777105767812362, 110.32608695652173,
#                  52.679245283018865, 9.152656621728786, 4.7851333032083145,
#                  8.437891632878731, 2.4620064899945917, 0.4034751151063363, 31.534942820838626]
wandb.config.is_loss_weights = 'no'
class_names = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
               'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
               'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present']

# criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weights).to(device))
labeled_criterion = torch.nn.BCEWithLogitsLoss()
unlabeled_criterion = UnlabeledLoss(t1=12, t2=17, alpha=3., class_weights=None)
# criterion = BCEwithLabelSmoothing(pos_weights=torch.tensor(class_weights).to(device))
# optimizer = Adas(model.parameters())
lr_start = 1e-4
lr_end = 1e-6
weight_decay = 0
epoch_num = 20
wandb.config.lr_start = lr_start
wandb.config.lr_end = lr_end
wandb.config.weight_decay = weight_decay
wandb.config.epoch_num = epoch_num
wandb.config.optimizer = 'adam'
wandb.config.scheduler = 'CosineAnnealingLR'

# optimizer = Adam(group_weight(model, weight_decay=weight_decay), lr=lr_start, weight_decay=0)
optimizer = Adam(model.parameters(), lr=lr_start, weight_decay=0)
scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=lr_end, last_epoch=-1)
model = model.to(device)
max_val_auc = 0

for epoch in range(epoch_num):
    train_loss, train_avg_auc, train_auc, train_rocs, train_data_pr, train_duration = train_one_epoch_pseudolabel(
        epoch, model, train_loader, unlabeled_loader, optimizer, labeled_criterion,
        unlabeled_criterion, device, scaler, iters_to_accumulate=accumulation_step, clip_grads=False
    )
    val_loss, val_avg_auc, val_auc, val_rocs, val_data_pr, val_duration = eval_model(
        model, val_loader, device, labeled_criterion, scaler)
    scheduler.step()

    writer.add_scalars('avg/loss', {'train': train_loss, 'val': val_loss}, epoch)
    writer.add_scalars('avg/auc', {'train': train_avg_auc, 'val': val_avg_auc}, epoch)
    for class_name, auc1, auc2 in zip(class_names, train_auc, val_auc):
        writer.add_scalars('AUC/{}'.format(class_name), {'train': auc1, 'val': auc2}, epoch)
    for i in range(len(class_names)):
        writer.add_pr_curve('PR curve train/{}'.format(class_names[i]),
                            train_data_pr[1][:, i], train_data_pr[0][:, i], global_step=epoch)
        writer.add_pr_curve('PR curve validation/{}'.format(class_names[i]),
                            val_data_pr[1][:, i], val_data_pr[0][:, i], global_step=epoch)
    writer.flush()

    for class_name, train_params, val_params in zip(class_names, train_rocs, val_rocs):
        train_data = [[x, y] for (x, y) in zip(*train_params)]
        wandb.log({'{} train_roc epoch {}'.format(class_name, epoch): wandb.Table(data=train_data, columns=['fpr', 'tpr'])})
        val_data = [[x, y] for (x, y) in zip(*val_params)]
        wandb.log({'{} val_roc epoch {}'.format(class_name, epoch): wandb.Table(data=val_data, columns=['fpr', 'tpr'])})
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
wandb.finish()
