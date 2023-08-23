# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午4:04
# @Author : zhuying
# @Company : Minivision
# @File : dataset_folder.py
# @Software : PyCharm

import cv2
import torch
import mxnet as mx
from torchvision import datasets
from torch.utils.data import Dataset
import numpy as np
import os
import sys
pwd = os.path.dirname(os.path.realpath(__file__))

try:
    from src.generate_patches import get_new_box
except Exception as e:
    sys.path.insert(1, os.path.join(pwd, "../.."))
    from src.generate_patches import get_new_box


def opencv_loader(path):
    img = cv2.imread(path)
    return img


class DatasetFolderFT(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 ft_width=10, ft_height=10, loader=opencv_loader):
        super(DatasetFolderFT, self).__init__(root, transform, target_transform, loader)
        self.root = root
        self.ft_width = ft_width
        self.ft_height = ft_height

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        # generate the FT picture of the sample
        ft_sample = generate_FT(sample)
        if sample is None:
            print('image is None --> ', path)
        if ft_sample is None:
            print('FT image is None -->', path)
        assert sample is not None

        ft_sample = cv2.resize(ft_sample, (self.ft_width, self.ft_height))
        ft_sample = torch.from_numpy(ft_sample).float()
        ft_sample = torch.unsqueeze(ft_sample, 0)

        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as err:
                print('Error Occured: %s' % err, path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, ft_sample, target


class MXDatasetFT(Dataset):
    def __init__(self, root, transform=None, target_transform=None,
                 ft_width=10, ft_height=10, scale=1.0):
        super(MXDatasetFT, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.ft_width = ft_width
        self.ft_height = ft_height
        self.scale = scale
        path_imgrec = os.path.join(root, f'train_{scale}.rec')
        path_imgidx = os.path.join(root, f'train_{scale}.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        # path, target = self.samples[index]
        # sample = self.loader(path)
        # # generate the FT picture of the sample
        # ft_sample = generate_FT(sample)
        # if sample is None:
        #     print('image is None --> ', path)
        # if ft_sample is None:
        #     print('FT image is None -->', path)
        # assert sample is not None
        
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        sample = mx.image.imdecode(img).asnumpy()  # RGB
        bbox = label[1:5].astype(np.int32)
        label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        
        # crop
        # bbox = get_new_box(sample.shape[0], sample.shape[1], bbox, scale=self.scale)
        # sample = sample[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
        ft_sample = generate_FT(sample)

        ft_sample = cv2.resize(ft_sample, (self.ft_width, self.ft_height))
        ft_sample = torch.from_numpy(ft_sample).float()
        ft_sample = torch.unsqueeze(ft_sample, 0)

        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as err:
                print('Error Occured: %s' % err, path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, ft_sample, label

    def __len__(self):
        return len(self.imgidx)

def generate_FT(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift)+1)
    maxx = -1
    minn = 100000
    for i in range(len(fimg)):
        if maxx < max(fimg[i]):
            maxx = max(fimg[i])
        if minn > min(fimg[i]):
            minn = min(fimg[i])
    fimg = (fimg - minn+1) / (maxx - minn+1)
    return fimg


if __name__ == "__main__":
    train_set = MXDatasetFT(root="/mnt/nvme0n1p2/datasets/untispoofing/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209", scale=1.5)
    print(train_set[0])