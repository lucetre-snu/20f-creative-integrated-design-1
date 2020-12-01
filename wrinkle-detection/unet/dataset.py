import os
import math
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.endswith('npy')]
        lst_input = [f for f in lst_data if f.endswith('jpg')]

        lst_label.sort()
        lst_input.sort()
        for i in range(len(lst_label)):
            if lst_label[i].split(".")[0] != lst_input[i].split(".")[0]:
                print("Error happened! image file doesn't match to label")
                print(lst_label[i])
                print(lst_input[i])

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.lst_input[index])).convert('RGB')
        input = np.array(img)
        label = np.load(os.path.join(self.data_dir, self.lst_label[index])).astype(np.uint8)
        label = label * 255

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data


## 트렌스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = input/255.0
        label = label/255.0

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        data = {'label': label, 'input': input}

        return data


class RandomResizedCrop(object):
    def __init__(self, ratio=0.3):
        self.ratio = ratio

    def __call__(self, data):
        label, input = data['label'], data['input']

        h, w = label.shape
        h_start = math.floor(np.random.rand(1)[0] * (h * self.ratio / 2.))
        h_end = h - math.floor(np.random.rand(1)[0] * (h * self.ratio / 2.))
        w_start = math.floor(np.random.rand(1)[0] * (w * self.ratio / 2.))
        w_end = w - math.floor(np.random.rand(1)[0] * (w * self.ratio / 2.))

        crop_label = label[h_start:h_end, w_start:w_end]
        crop_input = input[h_start:h_end, w_start:w_end, :]
        label = cv2.resize(crop_label, dsize=(w, h))
        input = cv2.resize(crop_input, dsize=(w, h))

        data = {'label': label, 'input': input}

        return data
