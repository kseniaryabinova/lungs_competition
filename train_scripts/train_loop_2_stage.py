import os
import shutil
import time
from datetime import datetime

from pytz import timezone

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':64:8'

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

from dataloader import ImageDataset, ImagesWithAnnotationsDataset
from efficient_net import EfficientNet3Stage
from train_functions import group_weight, CustomLoss, one_epoch_train_2_stage, \
    eval_model_2_stage

torch.manual_seed(25)
np.random.seed(25)

os.makedirs('tensorboard_runs', exist_ok=True)
shutil.rmtree('tensorboard_runs')
writer = SummaryWriter(log_dir='tensorboard_runs', filename_suffix=str(time.time()))

width_size = 640

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
        height=int(0.8192 * width_size),
        width=width_size,
        scale=(0.5, 1.5),
        p=0.5
    ),
    alb.ShiftScaleRotate(shift_limit=0.025, scale_limit=0.1, rotate_limit=20, p=0.5),
    alb.CoarseDropout(
        max_holes=12,
        min_holes=6,
        max_height=int(0.8192 * width_size / 6),
        max_width=int(width_size / 6),
        min_height=int(0.8192 * width_size / 20),
        min_width=int(width_size / 20),
        p=0.5
    ),
    alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
train_set = ImagesWithAnnotationsDataset(train_df, annot_df, train_image_transforms,
                                         '../ranzcr/train', width_size=width_size)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=48, pin_memory=True, drop_last=True)

val_df = df[df['fold'] == 1]
val_image_transforms = alb.Compose([
    alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
val_set = ImageDataset(val_df, val_image_transforms, '../../mark/ranzcr/train', width_size=width_size)
val_loader = DataLoader(val_set, batch_size=16, num_workers=48, pin_memory=True, drop_last=True)

checkpoints_dir_name = 'tf_efficientnet_b7_ns_640_2_stage_'
os.makedirs(checkpoints_dir_name, exist_ok=True)

teacher_model = EfficientNet3Stage(
    11, pretrained_backbone=True, mixed_precision=True, model_name='tf_efficientnet_b7_ns',
    checkpoint_path='tf_efficientnet_b7_ns_640_stage1/tf_efficientnet_b7_ns_640_stage1_epoch_4_val_auc_1.0_loss_'
                    '0.281_train_auc_0.992_loss_0.455.pth')
for param in teacher_model.parameters():
    param.requires_grad = False
student_model = EfficientNet3Stage(11, pretrained_backbone=True, mixed_precision=True,
                                   model_name='tf_efficientnet_b7_ns')

scaler = GradScaler()
if torch.cuda.device_count() > 1:
    teacher_model = torch.nn.DataParallel(teacher_model)
    student_model = torch.nn.DataParallel(student_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# class_weights = [354.625, 23.73913043478261, 2.777105767812362, 110.32608695652173,
#                  52.679245283018865, 9.152656621728786, 4.7851333032083145,
#                  8.437891632878731, 2.4620064899945917, 0.4034751151063363, 31.534942820838626]
class_names = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
               'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
               'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present']

train_criterion = CustomLoss(weights=(0.5, 1.))
valid_criterion = torch.nn.BCEWithLogitsLoss()

optimizer = Adam(group_weight(student_model, weight_decay=1e-4), lr=1e-4, weight_decay=0)
scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6, last_epoch=-1)
teacher_model = teacher_model.to(device)
student_model = student_model.to(device)
teacher_model.eval()

for epoch in range(20):
    total_train_loss, train_avg_auc, train_auc, train_data_pr, train_duration = one_epoch_train_2_stage(
        teacher_model, student_model, train_loader, optimizer, train_criterion,
        device, scaler, iters_to_accumulate=8, clip_grads=False)
    total_val_loss, val_avg_auc, val_auc, val_data_pr, val_duration = eval_model_2_stage(
        student_model, val_loader, device, valid_criterion, scaler)

    writer.add_scalars('avg/total loss', {'train': total_train_loss, 'val': total_val_loss}, epoch)
    writer.add_scalars('avg/auc', {'train': train_avg_auc, 'val': val_avg_auc}, epoch)
    for class_name, auc1, auc2 in zip(class_names, train_auc, val_auc):
        writer.add_scalars('AUC/{}'.format(class_name), {'train': auc1, 'val': auc2}, epoch)
    for i in range(len(class_names)):
        writer.add_pr_curve('PR curve train/{}'.format(class_names[i]),
                            train_data_pr[1][:, i], train_data_pr[0][:, i], global_step=epoch)
        writer.add_pr_curve('PR curve validation/{}'.format(class_names[i]),
                            val_data_pr[1][:, i], val_data_pr[0][:, i], global_step=epoch)
    writer.flush()

    print('EPOCH %d:\tTRAIN [duration %.3f sec, loss: %.3f, avg auc: %.3f]\t\t'
          'VAL [duration %.3f sec, loss: %.3f, avg auc: %.3f]\tCurrent time %s' %
          (epoch + 1, train_duration, total_train_loss, train_avg_auc,
           val_duration, total_val_loss, val_avg_auc, str(datetime.now(timezone('Europe/Moscow')))))

    torch.save(student_model.state_dict(),
               os.path.join(checkpoints_dir_name, '{}_epoch_{}_val_auc_{}_loss_{}_train_auc_{}_loss_{}.pth'.format(
                   checkpoints_dir_name, epoch + 1, round(val_avg_auc, 3), round(total_val_loss, 3),
                   round(train_avg_auc, 3), round(total_train_loss, 3))))

writer.close()
