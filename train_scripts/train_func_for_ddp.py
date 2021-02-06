import os
import shutil
import time
from datetime import datetime

import pandas as pd
import numpy as np

import torch

from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from albumentations.pytorch import ToTensorV2, ToTensor
import albumentations as alb

from adas_optimizer import Adas
from dataloader import ImageDataset
from efficient_net import EfficientNet
from train_functions import one_epoch_train, eval_model


def train_function(gpu,  world_size, node_rank, gpus):
    torch.manual_seed(25)
    np.random.seed(25)

    rank = node_rank * gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    width_size = 600

    if rank == 0:
        shutil.rmtree('tensorboard_runs', ignore_errors=True)
        writer = SummaryWriter(log_dir='tensorboard_runs', filename_suffix=str(time.time()))

    df = pd.read_csv('train_with_split.csv')
    train_df = df[df['split'] == 1]
    train_image_transforms = alb.Compose([
        alb.CLAHE(p=0.5),
        alb.GridDistortion(p=0.5),
        alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    train_set = ImageDataset(train_df, train_image_transforms, '../ranzcr/train', width_size=width_size)
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=5, shuffle=False,
                              num_workers=12, pin_memory=True, sampler=train_sampler)

    val_df = df[df['split'] == 0]
    val_image_transforms = alb.Compose([
        alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val_set = ImageDataset(val_df, val_image_transforms, '../ranzcr/train', width_size=width_size)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank)
    val_loader = DataLoader(val_set, batch_size=5, num_workers=12, pin_memory=True, sampler=val_sampler)

    checkpoints_dir_name = 'tf_efficientnet_b5'
    os.makedirs(checkpoints_dir_name, exist_ok=True)

    # model = ResNet18(11, 1, pretrained_backbone=True, mixed_precision=True)
    model = EfficientNet(11, pretrained_backbone=True, mixed_precision=True,
                         model_name='tf_efficientnet_b5_ns')
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
    model.to(device)
    scaler = GradScaler()
    model = DistributedDataParallel(model, device_ids=[gpu])

    class_weights = [354.625, 23.73913043478261, 2.777105767812362, 110.32608695652173,
                     52.679245283018865, 9.152656621728786, 4.7851333032083145,
                     8.437891632878731, 2.4620064899945917, 0.4034751151063363, 31.534942820838626]
    class_names = [
        'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
        'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
        'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present'
    ]
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weights).to(device))
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=1e-6, last_epoch=-1)

    for epoch in range(1, 30):
        total_train_loss, train_avg_auc, train_auc, train_duration = one_epoch_train(
            model, train_loader, optimizer, criterion, device, scaler)
        total_val_loss, val_avg_auc, val_auc, val_duration = eval_model(
            model, val_loader, device, criterion, scaler)

        if rank == 0:
            writer.add_scalars('avg/loss', {'train': total_train_loss, 'val': total_val_loss}, epoch)
            writer.add_scalars('avg/auc', {'train': train_avg_auc, 'val': val_avg_auc}, epoch)
            for class_name, auc1, auc2 in zip(class_names, train_auc, val_auc):
                writer.add_scalars('AUC/{}'.format(class_name), {'train': auc1, 'val': auc2}, epoch)

        print('EPOCH %d:\tTRAIN [duration %.3f sec, loss: %.3f, avg auc: %.3f]\t\t'
              'VAL [duration %.3f sec, loss: %.3f, avg auc: %.3f]\tCurrent time %s' %
              (epoch + 1, train_duration, total_train_loss, train_avg_auc,
               val_duration, total_val_loss, val_avg_auc, str(datetime.now())))
        print('{}\n{}'.format(str(train_auc), str(val_auc)))

        if rank == 0:
            torch.save(model.state_dict(),
                       os.path.join(checkpoints_dir_name,
                                    'model_epoch_{}_val_auc_{}_loss_{}_train_auc_{}_loss_{}.pth'.format(
                                        epoch + 1, round(val_avg_auc, 2), round(total_val_loss, 2),
                                        round(train_avg_auc, 2), round(total_train_loss, 2))))

    if rank == 0:
        writer.close()
