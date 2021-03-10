import os
import shutil
import time
from pytz import timezone
from datetime import datetime

import pandas as pd
import numpy as np

import torch

from torch.cuda.amp import GradScaler
from torch.nn import SyncBatchNorm
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from albumentations.pytorch import ToTensorV2, ToTensor
import albumentations as alb

from adas_optimizer import Adas
from dataloader import ImageDataset, NoisyStudentDataset, PseudolabelDataset
from efficient_net import EfficientNet, EfficientNetNoisyStudent
from train_functions import one_epoch_train, eval_model
from train_loop_noisy_student import ranzcr_train_df
from vit import ViT

import wandb


def train_function(gpu, world_size, node_rank, gpus):
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    torch.manual_seed(25)
    np.random.seed(25)

    rank = node_rank * gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    width_size = 640
    batch_size = 3
    accumulation_step = 10
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

    if rank == 0:
        wandb.init(project='effnet7', group=wandb.util.generate_id())
        wandb.config.width_size = width_size
        wandb.config.aspect_rate = 1
        wandb.config.batch_size = batch_size
        wandb.config.accumulation_step = accumulation_step

        shutil.rmtree('tensorboard_runs', ignore_errors=True)
        writer = SummaryWriter(log_dir='tensorboard_runs', filename_suffix=str(time.time()))

    ranzcr_df = pd.read_csv('test_pseudolabel.csv')
    padchest_df = pd.read_csv('padchest_pseudolabel.csv')

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
    train_set = PseudolabelDataset(ranzcr_train_df, padchest_df, train_image_transforms,
                                   '../rancor/test', 'data2', width_size=width_size)
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4, sampler=train_sampler)

    ranzcr_valid_df = pd.read_csv('train_folds.csv')
    ranzcr_valid_df = ranzcr_valid_df[ranzcr_valid_df['fold'] == 1]
    valid_image_transforms = alb.Compose([
        alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    valid_set = ImageDataset(ranzcr_valid_df, valid_image_transforms, '../ranzcr/train', width_size=width_size)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=4, pin_memory=False, drop_last=False)

    # ranzcr_valid_df = ranzcr_df[ranzcr_df['fold'] == 1]
    # valid_image_transforms = alb.Compose([
    #     alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #     ToTensorV2()
    # ])
    # valid_set = ImageDataset(ranzcr_valid_df, valid_image_transforms, '../ranzcr/train', width_size=width_size)
    # valid_sampler = DistributedSampler(valid_set, num_replicas=world_size, rank=rank)
    # valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=4, sampler=valid_sampler)

    checkpoints_dir_name = 'tf_efficientnet_b7_ns_pl_{}'.format(width_size)
    os.makedirs(checkpoints_dir_name, exist_ok=True)

    model = EfficientNetNoisyStudent(11, pretrained_backbone=True,
                                     mixed_precision=True, model_name='tf_efficientnet_b7_ns')
    model = SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)
    model = DistributedDataParallel(model, device_ids=[gpu])

    # class_weights = [354.625, 23.73913043478261, 2.777105767812362, 110.32608695652173,
    #                  52.679245283018865, 9.152656621728786, 4.7851333032083145,
    #                  8.437891632878731, 2.4620064899945917, 0.4034751151063363, 31.534942820838626]
    class_names = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
                   'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
                   'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present']
    scaler = GradScaler()
    criterion = torch.nn.BCEWithLogitsLoss()

    lr_start = 1e-6
    lr_end = 1e-7
    weight_decay = 0
    epoch_num = 10
    if rank == 0:
        wandb.config.model_name = checkpoints_dir_name
        wandb.config.lr_start = lr_start
        wandb.config.lr_end = lr_end
        wandb.config.weight_decay = weight_decay
        wandb.config.epoch_num = epoch_num
        wandb.config.optimizer = 'adam'
        wandb.config.scheduler = 'CosineAnnealingLR'
        wandb.config.is_loss_weights = 'no'

    optimizer = Adam(model.parameters(), lr=lr_start, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=lr_end, last_epoch=-1)

    max_val_auc = 0

    for epoch in range(epoch_num):
        train_loss, train_avg_auc, train_auc, train_rocs, train_data_pr, train_duration = one_epoch_train(
            model, train_loader, optimizer, criterion, device, scaler,
            iters_to_accumulate=accumulation_step, clip_grads=False)
        scheduler.step()

        if rank == 0:
            val_loss, val_avg_auc, val_auc, val_rocs, val_data_pr, val_duration = eval_model(
                model, valid_loader, device, criterion, scaler)

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
    if rank == 0:
        wandb.finish()
