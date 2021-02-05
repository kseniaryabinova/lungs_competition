import glob
import math
import os
import time
from itertools import chain, cycle
from typing import Iterator

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from albumentations.pytorch import ToTensorV2, ToTensor
import albumentations as alb

from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch
from torch.utils.data.dataset import T_co

from torchvision import transforms

torch.manual_seed(25)


class ImageIterableDataset(IterableDataset):
    def __init__(self, df: pd.DataFrame, batch_size, transform,
                 dataset_filepath, image_h_w_ratio=0.8192, width_size=128):
        self.df = df
        self.ids = list(range(len(self.df)))
        self.batch_size = batch_size
        self.transform = transform
        self.dataset_filepath = dataset_filepath
        self.image_h_w_ratio = image_h_w_ratio
        self.width_size = width_size
        self.height_size = int(self.image_h_w_ratio * self.width_size)

    def get_sample(self, image_id):
        image_name = '{}.jpg'.format(self.df.iloc[image_id]['StudyInstanceUID'])
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

        labels = self.df.iloc[image_id, 1:12].values.astype('float').reshape(11)

        return result_image['image'], labels

    def process_data(self, data):
        # for image_id in data:
        worker = torch.utils.data.get_worker_info()
        worker_id = id(self) if worker is not None else -1

        yield self.get_sample(data)

    def get_stream(self):
        return chain.from_iterable(map(self.process_data, cycle(self.ids)))

    def get_streams(self):
        return zip(*[self.get_stream() for _ in range(self.batch_size)])

    def __iter__(self):
        return self.get_streams()

    @classmethod
    def split_datasets(cls, data_list, batch_size, max_workers, transform,
                 dataset_filepath, image_h_w_ratio=0.8192, width_size=128):

        num_workers = 1
        for n in range(max_workers, 0, -1):
            if batch_size % n == 0:
                num_workers = n
                break

        split_size = batch_size // num_workers

        return [cls(data_list, split_size, transform,
                 dataset_filepath, image_h_w_ratio, width_size) for _ in range(num_workers)]


class MultiStreamDataLoader:
    def __init__(self, datasets):
        self.datasets = datasets

    def get_stream_loaders(self):
        return zip(*[DataLoader(dataset, num_workers=1, batch_size=None) for dataset in self.datasets])

    def __iter__(self):
        for batch_parts in self.get_stream_loaders():
            yield list(chain(*batch_parts))


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
            height=int(0.8192*640),
            width=640,
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
            max_height=int(0.8192*640 / 6),
            max_width=int(640 / 6),
            min_height=int(0.8192*640 / 20),
            min_width=int(640 / 20),
            p=0.5
        ),
        alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    dataset = ImageDataset(train_df, image_transforms, '../dataset/train', width_size=640)

    for i in range(50, 60):
        image, labels = dataset[i]
        cv2.imshow('1', image)
        if cv2.waitKey(0) == 27:
            break

    # dataset = ImageIterableDataset(train_df, 6400, image_transforms, '../ranzcr/train', max_workers=10)
    # datasets = ImageIterableDataset.split_datasets(train_df, 6400, 24, image_transforms, '../ranzcr/train')
    # dataloader = MultiStreamDataLoader(datasets)
    # for data in dataloader:
    #     start_time = time.time()
    #
    #     print(len(data))
    #     print(len(data[0]))
    #     print(len(data[0][0]))
    #     print(len(data[0]), type(data[0]))
    #     print(data[0][0][0].shape, type(data[0][0][0]))
    #     print(data[0][0][1].shape, type(data[0][0][1]))
    #
    #     print(time.time() - start_time)
