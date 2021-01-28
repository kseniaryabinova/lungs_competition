import glob
import os

import cv2
import pandas as pd
import numpy as np

from albumentations.pytorch import ToTensorV2
import albumentations as alb

from torch.utils.data import Dataset
import torch

from torchvision import transforms

torch.manual_seed(25)


class ImageDataset(Dataset):
    def __init__(self, df, transform, dataset_filepath, image_h_w_ratio=0.8192, width_size=128):
        self.df = df
        self.transform = transform
        self.dataset_filepath = dataset_filepath
        self.image_h_w_ratio = image_h_w_ratio
        self.width_size = width_size
        self.height_size = int(self.image_h_w_ratio * self.width_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = '{}.jpg'.format(self.df.iloc[idx]['StudyInstanceUID'])
        image_filepath = os.path.join(self.dataset_filepath, image_name)
        # image = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(image_filepath)

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

        result_image = np.full((self.height_size, self.width_size, 3), 0, dtype=np.uint8)
        result_image[t_padding:b_padding, l_padding:r_padding, :] = image
        result_image = np.reshape(result_image, (result_image.shape[0], result_image.shape[1], 3))

        if self.transform:
            result_image = self.transform(image=result_image)

        labels = self.df.iloc[idx, 1:12].values.astype('float').reshape(11)

        return result_image['image'], labels


if __name__ == '__main__':
    train_df = pd.read_csv('../dataset/train.csv')

    image_transforms = alb.Compose([
        alb.CLAHE(),
        # alb.RandomBrightnessContrast(p=1.0),
        # alb.ElasticTransform(sigma=8, alpha_affine=8, p=1.0),
        alb.GridDistortion(p=1),
        ToTensorV2(),
    ])

    dataset = ImageDataset(train_df, image_transforms, '../dataset/train')

    for i in range(50, 60):
        image, labels = dataset[i]
        cv2.imshow('1', image)
        cv2.waitKey(0)

