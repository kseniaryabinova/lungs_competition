import os

import torch
import pandas as pd
from albumentations.pytorch import ToTensorV2
import albumentations as alb
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from dataloader import ImageDataset
from efficient_net import EfficientNetNoisyStudent
from train_functions import eval_model

checkpoint_paths = [
    'tf_efficientnet_b7_noisy_student_640/tf_efficientnet_b7_noisy_student_640_epoch17_val_auc0.964_loss0.12_train_auc0.894_loss0.156.pth',
    'tf_efficientnet_b7_noisy_student_640/tf_efficientnet_b7_noisy_student_640_epoch18_val_auc0.964_loss0.12_train_auc0.897_loss0.155.pth',
    'tf_efficientnet_b7_noisy_student_640/tf_efficientnet_b7_noisy_student_640_epoch19_val_auc0.965_loss0.119_train_auc0.901_loss0.154.pth',
    'tf_efficientnet_b7_noisy_student_640/tf_efficientnet_b7_noisy_student_640_epoch20_val_auc0.965_loss0.119_train_auc0.899_loss0.155.pth',
]

state_dicts = []
for checkpoint_path in checkpoint_paths:
    model = EfficientNetNoisyStudent(11, pretrained_backbone=True, mixed_precision=True,
                                     model_name='tf_efficientnet_b7_ns', checkpoint_path=checkpoint_path)
    model.eval()
    model = model.float()
    state_dicts.append(model.state_dict())

avg_model = EfficientNetNoisyStudent(11, pretrained_backbone=True, mixed_precision=True,
                                     model_name='tf_efficientnet_b7_ns')
avg_state_dict = avg_model.state_dict()

for key in avg_state_dict:
    avg_state_dict[key] = torch.zeros(avg_state_dict[key].shape)
    for state_dict in state_dicts:
        avg_state_dict[key] += state_dict[key]
    avg_state_dict[key] = avg_state_dict[key] / float(len(state_dicts))

avg_model.load_state_dict(avg_state_dict)

ranzcr_df = pd.read_csv('train_folds.csv')
ranzcr_valid_df = ranzcr_df[ranzcr_df['fold'] == 1]
valid_image_transforms = alb.Compose([
    alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
valid_set = ImageDataset(ranzcr_valid_df, valid_image_transforms, '../ranzcr/train', width_size=640)
valid_loader = DataLoader(valid_set, batch_size=12, num_workers=12, pin_memory=False, drop_last=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.BCEWithLogitsLoss()

scaler = GradScaler()
if torch.cuda.device_count() > 1:
    avg_model = torch.nn.DataParallel(avg_model)
avg_model = avg_model.to(device)

val_loss, val_avg_auc, val_auc, val_rocs, val_data_pr, val_duration = eval_model(
    avg_model, valid_loader, device, criterion, scaler)

torch.save(avg_model.module.state_dict(),
           os.path.join('effnet7_wa_val_auc{}_loss{}.pth'.format(
               round(val_avg_auc, 3), round(val_loss, 3))))
