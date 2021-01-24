import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import pandas as pd

from resnet import ResNet18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ResNet18(11, 1, pretrained_backbone=False, mixed_precision=False)
model.load_state_dict(torch.load('model.pth'))
model.eval()
model.to(device)


class ValImageDataset(Dataset):
    def __init__(self, transform, dataset_filepath, image_h_w_ratio=0.8192, width_size=128):
        self.files = [filepath for filepath in glob.iglob(os.path.join(dataset_filepath, '*'))]
        self.transform = transform
        self.image_h_w_ratio = image_h_w_ratio
        self.width_size = width_size
        self.height_size = int(self.image_h_w_ratio * self.width_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_filepath = self.files[idx]
        image = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)

        image_h, image_w = image.shape[0], image.shape[1]
        if image_h / image_w > self.image_h_w_ratio:
            ratio_coeff = self.height_size / image_h
        else:
            ratio_coeff = self.width_size / image_w
        new_h = int(image_h * ratio_coeff)
        new_w = int(image_w * ratio_coeff)
        image = cv2.resize(image, (new_w, new_h))

        w_padding = (self.width_size - new_w) / 2
        h_padding = (self.height_size - new_h) / 2
        l_padding = int(w_padding)
        t_padding = int(h_padding)
        r_padding = int(new_w + (w_padding if w_padding % 1 == 0 else w_padding - 0.5))
        b_padding = int(new_h + (h_padding if h_padding % 1 == 0 else h_padding - 0.5))

        result_image = np.full((self.height_size, self.width_size), 0, dtype=np.uint8)
        result_image[t_padding:b_padding, l_padding:r_padding] = image

        if self.transform:
            result_image = self.transform(result_image)

        return image_filepath.rsplit('/', 1)[1].rsplit('.', 1)[0], result_image


image_transforms = transforms.Compose([
    transforms.ToTensor(),
])
test_set = ValImageDataset(image_transforms, '../dataset/test_test')
test_loader = DataLoader(test_set, batch_size=32, num_workers=12, pin_memory=True)

predictions = []
filepaths = []

with torch.no_grad():
    for batch in test_loader:
        filepaths_batch, image_batch = batch
        prediction_batch = model(image_batch.to(device))

        predictions.extend(prediction_batch.cpu().numpy())
        filepaths.extend(filepaths_batch)

predictions = np.array(predictions)
df = pd.DataFrame({
    'StudyInstanceUID': filepaths,
    'ETT - Abnormal': predictions[:, 0],
    'ETT - Borderline': predictions[:, 1],
    'ETT - Normal': predictions[:, 2],
    'NGT - Abnormal': predictions[:, 3],
    'NGT - Borderline': predictions[:, 4],
    'NGT - Incompletely Imaged': predictions[:, 5],
    'NGT - Normal': predictions[:, 6],
    'CVC - Abnormal': predictions[:, 7],
    'CVC - Borderline': predictions[:, 8],
    'CVC - Normal': predictions[:, 9],
    'Swan Ganz Catheter Present': predictions[:, 10]})
df.to_csv('submission.csv', index=False)
