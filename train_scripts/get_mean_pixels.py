import pandas as pd
import numpy as np
import cv2

import os
import time


df = pd.read_csv('../dataset/train_with_split.csv')

channel = 0
total = 0

for i, idx in enumerate(df[df['split'] == 1]['StudyInstanceUID']):
    if i % 2000 == 0:
        print(i+1, time.time())

    filepath = os.path.join('../dataset/train', '{}.jpg'.format(idx))
    img = cv2.imread(filepath)
    total += img.shape[0] * img.shape[1]
    channel += np.sum(img)

channel_mean = channel / total
print(channel_mean)
channel = 0

for i, idx in enumerate(df[df['split'] == 1]['StudyInstanceUID']):
    if i % 2000 == 0:
        print(i+1, time.time())

    filepath = os.path.join('../dataset/train', '{}.jpg'.format(idx))
    img = cv2.imread(filepath)

    channel += np.sum((img - channel_mean) ** 2)

channel_std = np.sqrt(channel / total)
print(channel_std)
