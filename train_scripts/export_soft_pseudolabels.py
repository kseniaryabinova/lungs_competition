import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import pandas as pd
import numpy as np

import torch

from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from albumentations.pytorch import ToTensorV2
import albumentations as alb

from dataloader import ChestXDataset
from efficient_net import EfficientNet, EfficientNetB5

torch.manual_seed(25)
np.random.seed(25)

width_size = 640

df = pd.read_csv('data.csv')
image_transforms = alb.Compose([
    alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
dataset = ChestXDataset(df, image_transforms, '../data', width_size=width_size)
loader = DataLoader(dataset, batch_size=64, num_workers=48, pin_memory=True, drop_last=False, shuffle=False)

model = EfficientNetB5(model_name='tf_efficientnet_b5_ns')
model.load_state_dict(torch.load('tf_efficientnet_b5_ns_CV96.21.pth')['model'])

scaler = GradScaler()
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

predictions = []
sigmoid = torch.nn.Sigmoid()
iter_counter = 0
start_time = time.time()

model.eval()
model = model.float()

with torch.no_grad():
    for i, batch in enumerate(loader):
        inputs, labels = batch
        with autocast():
            outputs = model(inputs.to(device))

        predictions.extend(sigmoid(outputs).cpu().detach().numpy())

        if i % 100 == 0:
            print(i * 64)

predictions = np.array(predictions, dtype=np.float)

class_names = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
               'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
               'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present']

for i, class_name in enumerate(class_names):
    df[class_name] = predictions[:, i]

df.to_csv('chestx_pseudolabeled_data.csv', index=False)
