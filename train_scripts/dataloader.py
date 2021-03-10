import ast
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

np.random.seed(25)
torch.manual_seed(25)

COLOR_MAP = {'ETT - Abnormal': (255, 0, 0),
             'ETT - Borderline': (0, 255, 0),
             'ETT - Normal': (0, 0, 255),
             'NGT - Abnormal': (255, 255, 0),
             'NGT - Borderline': (255, 0, 255),
             'NGT - Incompletely Imaged': (0, 255, 255),
             'NGT - Normal': (128, 0, 0),
             'CVC - Abnormal': (0, 128, 0),
             'CVC - Borderline': (0, 0, 128),
             'CVC - Normal': (128, 128, 0),
             'Swan Ganz Catheter Present': (128, 0, 128),
             }


class ImagesWithAnnotationsDataset(Dataset):
    def __init__(self, df, df_annot, transform, dataset_filepath, image_h_w_ratio=0.8192, width_size=128):
        self.df = df
        self.df_annot = df_annot
        self.filenames = self.df[self.df['StudyInstanceUID'].isin(self.df_annot['StudyInstanceUID'])][
            'StudyInstanceUID'].unique().tolist()

        self.transform = transform
        self.dataset_filepath = dataset_filepath
        self.image_h_w_ratio = image_h_w_ratio
        self.width_size = width_size
        self.height_size = int(self.image_h_w_ratio * self.width_size)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_name = '{}.jpg'.format(self.filenames[idx])
        image_filepath = os.path.join(self.dataset_filepath, image_name)
        image = cv2.imread(image_filepath)
        image_wo_annot = image.copy()

        df_patient = self.df_annot[self.df_annot["StudyInstanceUID"] == self.filenames[idx]]
        if df_patient.shape[0]:
            labels = df_patient["label"].values.tolist()
            lines = df_patient["data"].apply(ast.literal_eval).values.tolist()
            for line, label in zip(lines, labels):
                for x, y in line:
                    cv2.circle(image, (x, y), radius=40, color=COLOR_MAP[label], thickness=-1)

        image_h, image_w = image.shape[0], image.shape[1]
        if image_h / image_w > self.image_h_w_ratio:
            ratio_coeff = self.height_size / image_h
        else:
            ratio_coeff = self.width_size / image_w
        new_h = int(image_h * ratio_coeff)
        new_w = int(image_w * ratio_coeff)
        image = cv2.resize(image, (new_w, new_h))
        image_wo_annot = cv2.resize(image_wo_annot, (new_w, new_h))

        w_padding = (self.width_size - new_w) / 2
        h_padding = (self.height_size - new_h) / 2
        l_padding = int(w_padding)
        t_padding = int(h_padding)
        r_padding = int(new_w + (w_padding if w_padding % 1 == 0 else w_padding - 0.5))
        b_padding = int(new_h + (h_padding if h_padding % 1 == 0 else h_padding - 0.5))

        result_image = np.full((self.height_size, self.width_size, 3), 0, dtype=np.uint8)
        result_image[t_padding:b_padding, l_padding:r_padding, :] = image
        result_image = np.reshape(result_image, (result_image.shape[0], result_image.shape[1], 3))
        result_image_wo_annot = np.full((self.height_size, self.width_size, 3), 0, dtype=np.uint8)
        result_image_wo_annot[t_padding:b_padding, l_padding:r_padding, :] = image_wo_annot
        result_image_wo_annot = np.reshape(result_image_wo_annot,
                                           (result_image_wo_annot.shape[0], result_image_wo_annot.shape[1], 3))

        image_df = self.df[self.df['StudyInstanceUID'] == self.filenames[idx]]
        labels = image_df.iloc[0, 1:12].values.astype('float').reshape(11)

        if self.transform:
            result_image = self.transform(image=result_image)
            result_image_wo_annot = self.transform(image=result_image_wo_annot)

        return result_image['image'], result_image_wo_annot['image'], labels


class ChestXDataset(Dataset):
    def __init__(self, df, transform, dataset_filepath, width_size=128):
        self.df = df
        self.transform = transform
        self.dataset_filepath = dataset_filepath
        self.width_size = width_size
        self.classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
                        'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'No Finding', 'Mass',
                        'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_file = self.df.iloc[idx]['Image Index']
        filepath = os.path.join(self.dataset_filepath, self.df.iloc[idx]['dirname'], 'images', image_file)
        image = cv2.imread(filepath)
        image = cv2.resize(image, (self.width_size, self.width_size))

        if self.transform:
            image = self.transform(image=image)['image']

        labels = self.df.iloc[idx][self.classes].values.astype('float').reshape(len(self.classes))

        return image, labels


class NoisyStudentDataset(Dataset):
    def __init__(self, ranzcr_df, chestx_df, transform, ranzcr_path, chestx_path, width_size=128):
        self.ranzcr_df = ranzcr_df
        self.chestx_df = chestx_df
        self.transform = transform
        self.ranzcr_path = ranzcr_path
        self.chestx_path = chestx_path
        self.width_size = width_size

        self.indices = np.arange(len(self.ranzcr_df) + len(self.chestx_df))
        np.random.shuffle(self.indices)

        self.class_names = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
                            'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
                            'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present']

    def __len__(self):
        return len(self.ranzcr_df) + len(self.chestx_df)

    def __getitem__(self, idx):
        index = self.indices[idx]

        if index < len(self.ranzcr_df):
            image_name = '{}.jpg'.format(self.ranzcr_df.iloc[index]['StudyInstanceUID'])
            image_filepath = os.path.join(self.ranzcr_path, image_name)
            image = cv2.imread(image_filepath)
            labels = self.ranzcr_df.iloc[index, 1:12].values.astype('float').reshape(len(self.class_names))
        else:
            index -= len(self.ranzcr_df)
            image_file = self.chestx_df.iloc[index]['Image Index']
            filepath = os.path.join(self.chestx_path, self.chestx_df.iloc[index]['dirname'], 'images', image_file)
            image = cv2.imread(filepath)
            labels = self.chestx_df.iloc[index][self.class_names].values.astype('float').reshape(len(self.class_names))

        image = cv2.resize(image, (self.width_size, self.width_size))
        if self.transform:
            image = self.transform(image=image)['image']

        return image, labels


class PseudolabelDataset(Dataset):
    def __init__(self, ranzcr_df, padchest_df, transform, ranzcr_path, padchest_path, width_size=128):
        self.ranzcr_df = ranzcr_df
        self.padchest_df = padchest_df
        self.transform = transform
        self.ranzcr_path = ranzcr_path
        self.padchest_path = padchest_path
        self.width_size = width_size

        self.indices = np.arange(len(self.ranzcr_df) + len(self.padchest_df))
        np.random.shuffle(self.indices)

        self.class_names = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
                            'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
                            'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present']

    def __len__(self):
        return len(self.ranzcr_df) + len(self.padchest_df)

    def __getitem__(self, idx):
        index = self.indices[idx]

        if index < len(self.ranzcr_df):
            image_name = self.ranzcr_df.iloc[index]['filepaths']
            image_filepath = os.path.join(image_name)
            image = cv2.imread(image_filepath)
            labels = self.ranzcr_df.iloc[index][self.class_names].values.astype('float').reshape(len(self.class_names))
        else:
            index -= len(self.ranzcr_df)
            filepath = self.padchest_df.iloc[index]['filepaths']
            image = cv2.imread(filepath)
            labels = self.padchest_df.iloc[index][self.class_names].values.astype('float').reshape(len(self.class_names))

        image = cv2.resize(image, (self.width_size, self.width_size))
        if self.transform:
            image = self.transform(image=image)['image']

        return image, labels


class UnlabeledImageDataset(Dataset):
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
        image = cv2.imread(image_filepath)
        image = cv2.resize(image, (self.width_size, self.width_size))

        if self.transform:
            image = self.transform(image=image)['image']

        return image, image_filepath


class PadChestImageDataset(Dataset):
    def __init__(self, df, transform, dataset_filepath, image_h_w_ratio=0.8192, width_size=128):
        self.df = df
        self.dataset_filepath = dataset_filepath
        self.transform = transform
        self.image_h_w_ratio = image_h_w_ratio
        self.width_size = width_size
        self.height_size = int(self.image_h_w_ratio * self.width_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_file = self.df.iloc[idx]['ImageID']
        filepath = os.path.join(self.dataset_filepath, image_file)
        image = cv2.imread(filepath)
        image = cv2.resize(image, (self.width_size, self.width_size))

        if self.transform:
            image = self.transform(image=image)['image']

        return image, filepath


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

        # image_h, image_w = image.shape[0], image.shape[1]
        # if image_h / image_w > self.image_h_w_ratio:
        #     ratio_coeff = self.height_size / image_h
        # else:
        #     ratio_coeff = self.width_size / image_w
        # new_h = int(image_h * ratio_coeff)
        # new_w = int(image_w * ratio_coeff)
        # image = cv2.resize(image, (new_w, new_h))

        # w_padding = (self.width_size - new_w) / 2
        # h_padding = (self.height_size - new_h) / 2
        # l_padding = int(w_padding)
        # t_padding = int(h_padding)
        # r_padding = int(new_w + (w_padding if w_padding % 1 == 0 else w_padding - 0.5))
        # b_padding = int(new_h + (h_padding if h_padding % 1 == 0 else h_padding - 0.5))
        #
        # result_image = np.full((self.height_size, self.width_size, 3), 0, dtype=np.uint8)
        # result_image[t_padding:b_padding, l_padding:r_padding, :] = image
        # result_image = np.reshape(result_image, (result_image.shape[0], result_image.shape[1], 3))

        result_image = cv2.resize(image, (self.width_size, self.width_size))

        if self.transform:
            result_image = self.transform(image=result_image)

        labels = self.df.iloc[idx, 1:12].values.astype('float').reshape(11)

        return result_image['image'], labels


if __name__ == '__main__':
    train_df = pd.read_csv('../dataset/train.csv')
    annot_df = pd.read_csv('../dataset/train_annotations.csv')
    width_size = 600

    image_transforms = alb.Compose([
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
            # height=int(0.8192 * width_size),
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
    ])

    dataset = ImageDataset(train_df, image_transforms, '../dataset/train', width_size=640)
    # dataset = ImagesWithAnnotationsDataset(train_df, annot_df, image_transforms,
    #                                        '../dataset/train', width_size=width_size)

    for i in range(10):
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
