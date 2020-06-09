"""
https://github.com/gidariss/FewShotWithoutForgetting/blob/master/dataloader.py
"""
import os
import sys
import time
import tqdm
import random
import numpy as np
from PIL.Image import BILINEAR

import torch
from torchvision import transforms
# from Test.test_Cutout import Cutout
from torchvision.transforms import functional as Ft
import torchnet as tnt
from compare.autoaugment import ImageNetPolicy
sys.dont_write_bytecode = True


# Batch(list terms:5):
# support_images: 8x5x3x84x84
# support_labels: 8x5
# query_images: 8x30x3x84x84
# query_labels: 8x30
# 8x5
# 8

def Rotate(angle):
    """Random rotate PIL Image
    :param angle:
    :return:
    """
    def rotate_img(img, angle=angle):
        img = Ft.rotate(img, angle, resample=BILINEAR)
        return img
    return rotate_img


rotation_transforms = []
for i in range(4):
    angle = i * 90
    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.RandomCrop(84, padding=8),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        Rotate(angle),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    rotation_transforms.append(transform)

AutoAugment_Transform = transforms.Compose([
    # transforms.Resize((256, 256)),
    # ImageNetPolicy(),
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class FewShotDataloader(object):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 epoch_size=2000,
                 num_workers=4,
                 shuffle=True,
                 ):
        self.dataset = dataset
        self.data_name = dataset.data_name
        self.data_root = dataset.data_root  # the root file to store images
        if self.data_name in ['miniImageNet', 'tieredImageNet', 'CUB']:
            self.data_root = os.path.join(self.data_root, 'images')
        self.data_list = dataset.data_list
        self.way_num = dataset.way_num
        self.shot_num = 1  # default: 1
        self.query_num = dataset.query_num
        self.transform = dataset.transform
        self.query_transform = AutoAugment_Transform
        self.rotation_transform = rotation_transforms
        self.loader = dataset.loader
        self.n_s_augment = 1
        self.n_q_augment = 4  # default:5

        # self.query_transform = AutoAugment_Transform

        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def sample_episode(self):
        """Samples a training episode."""
        way_num = self.way_num
        shot_num = self.shot_num
        Support_imgs = random.sample(self.data_list, way_num * shot_num)
        Support_imgs = [(img_name, label) for label, (img_name, _) in enumerate(Support_imgs)]
        return Support_imgs

    def createExamplesTensorData(self, examples):
        images = torch.stack([self.transform(self.loader(os.path.join(self.data_root, img_name)))
                              for img_name, _ in examples for _ in range(self.n_s_augment)], dim=0)
        labels = torch.tensor([label for _, label in examples for _ in range(self.n_s_augment)]).long()
        return images, labels

    def createQueryTensorData(self, examples):
        images = torch.stack([self.query_transform(self.loader(os.path.join(self.data_root, img_name)))
                              for img_name, _ in examples for _ in range(self.n_q_augment)], dim=0)
        labels = torch.tensor([label for _, label in examples for _ in range(self.n_q_augment)]).long()
        return images, labels

    def createRotateTensorData(self, examples):
        images = torch.stack([rotate_transform(self.loader(os.path.join(self.data_root, img_name)))
                              for img_name, _ in examples for rotate_transform in self.rotation_transform], dim=0)
        labels = torch.tensor([label for _, _ in examples for label in range(len(self.rotation_transform))]).long()
        return images, labels

    def get_iterator(self, epoch=0):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        def load_function(iter_idx):
            Support_imgs = self.sample_episode()
            Xt, Yt = self.createQueryTensorData(Support_imgs)
            Xr, Yr = self.createRotateTensorData(Support_imgs)
            Xe, Ye = self.createExamplesTensorData(Support_imgs)
            return Xt, Yt, Xr, Yr, Xe, Ye

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=load_function)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle)

        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return int(self.epoch_size / self.batch_size)


if __name__ == '__main__':
    from engine.configs.parser import BaseOptions
    from datasets.few_shot_dataset import FewShotDataSet
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    opts = BaseOptions().opts
    train_dataset = FewShotDataSet(opts, phase='train', transform=transform)
    data_list = dict(train_dataset.data_list)
    train_loader = FewShotDataloader(train_dataset, batch_size=8, epoch_size=10000, num_workers=0)

    start_time = time.time()
    for i, batch in enumerate(tqdm.tqdm(train_loader(0)), 1):
        query_imgs, query_labels, rotate_imgs, rotate_labels, support_imgs, support_labels = batch

        print(query_imgs.size(), query_labels.size())  # ([8, 5, 3, 84, 84]) torch.Size([8, 5])
        print(rotate_imgs.size(), rotate_labels.size())
        print(support_imgs.size(), support_labels.size())  # ([8, 5, 3, 84, 84]) torch.Size([8, 5])
        # print(support_labels)

        from preprocess.visualize import show_Images
        show_Images(support_imgs[0], 5, 'Support Images')
        show_Images(query_imgs[0], 5, 'Query Images')
        show_Images(rotate_imgs[0], 4, 'Query Images')

    end_time = time.time()
    print('Cost time:{}'.format(end_time - start_time))

    # Cost time: 195.35s
    # Cost time: 1498.60
