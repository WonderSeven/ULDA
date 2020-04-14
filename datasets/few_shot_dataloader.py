"""
https://github.com/gidariss/FewShotWithoutForgetting/blob/master/dataloader.py
"""
import os
import sys
import time
import tqdm
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
import torchnet as tnt
sys.dont_write_bytecode = True

# Batch(list terms:5):
# support_images: 8x5x3x84x84
# support_labels: 8x5
# query_images: 8x30x3x84x84
# query_labels: 8x30
# 8x5
# 8

class FewShotDataloader(object):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 epoch_size=2000,
                 num_workers=4,
                 shuffle=True,
                 ):
        self.dataset     = dataset
        self.data_name   = dataset.data_name
        self.data_root   = dataset.data_root  # the root file to store images
        if self.data_name in ['miniImageNet', 'tieredImageNet', 'CUB']:
            self.data_root = os.path.join(self.data_root, 'images')
        self.way_num     = dataset.way_num
        self.shot_num    = dataset.shot_num
        self.query_num   = dataset.query_num
        self.transform   = dataset.transform
        self.loader      = dataset.loader

        self.batch_size  = batch_size
        self.epoch_size  = epoch_size
        self.num_workers = num_workers
        self.shuffle     = shuffle

    def sampleImageIdsFrom(self, cat_id, sample_size=1):
        assert (cat_id in self.dataset.class_img_dict)
        assert (len(self.dataset.class_img_dict[cat_id]) >= sample_size)
        # Note: random.sample samples elements without replacement.
        return random.sample(self.dataset.class_img_dict[cat_id], sample_size)

    def sampleCategories(self, sample_size=1):
        class_list = self.dataset.class_list
        assert (len(class_list) >= sample_size)
        return random.sample(class_list, sample_size)

    def sample_query_examples(self, categories, query_num):
        Tbase = []
        if len(categories) > 0:
            # average sample (keep the number of images each category)
            for K_idx, K_cat in enumerate(categories):
                img_ids = self.sampleImageIdsFrom(
                    K_cat, sample_size=query_num)
                Tbase += [(img_id, K_idx) for img_id in img_ids]

        assert(len(Tbase)) == len(categories) * query_num
        return Tbase

    def sample_support_and_query_examples(
            self, categories, query_num, shot_num):
        if len(categories) == 0:
            return [], []
        nCategories = len(categories)
        Query_imgs = []
        Support_imgs = []

        for idx in range(len(categories)):
            img_ids = self.sampleImageIdsFrom(
                categories[idx],
                sample_size=(query_num + shot_num)
            )
            imgs_novel = img_ids[:query_num]
            imgs_exemplar = img_ids[query_num:]

            Query_imgs += [(img_id, idx) for img_id in imgs_novel]
            Support_imgs += [(img_id, idx) for img_id in imgs_exemplar]

        assert(len(Query_imgs) == nCategories * query_num)
        assert(len(Support_imgs) == nCategories * shot_num)

        return Query_imgs, Support_imgs

    def sample_episode(self):
        """Samples a training episode."""
        way_num  = self.way_num
        shot_num  = self.shot_num
        query_num = self.query_num
        categories = self.sampleCategories(way_num)
        Query_imgs, Support_imgs = self.sample_support_and_query_examples(categories, query_num, shot_num)
        return Query_imgs, Support_imgs

    def createExamplesTensorData(self, examples):
        images = torch.stack([self.transform(self.loader(os.path.join(self.data_root, img_name))) for img_name, _ in examples], dim=0)
        labels = torch.tensor([label for _, label in examples]).long()
        return images, labels

    def get_iterator(self, epoch=0):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        def load_function(iter_idx):
            Query_imgs, Support_imgs = self.sample_episode()
            Xt, Yt = self.createExamplesTensorData(Query_imgs)
            Xe, Ye = self.createExamplesTensorData(Support_imgs)
            return Xt, Yt, Xe, Ye

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
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    opts = BaseOptions().opts
    train_dataset = FewShotDataSet(opts, phase='train', transform=transform)

    train_loader = FewShotDataloader(train_dataset, batch_size=8, epoch_size=10000)
    categories = train_loader.sampleCategories(5)
    print('Sampled categories:{}'.format(categories))
    one_category = categories[0]
    imgs = train_loader.sampleImageIdsFrom(one_category, 5)
    print('Sampled images:{}'.format(imgs))
    Query_imgs, Support_imgs = train_loader.sample_support_and_query_examples(categories, 15, 1)

    images, labels = train_loader.createExamplesTensorData(Support_imgs)

    start_time = time.time()
    for i, batch in enumerate(tqdm.tqdm(train_loader(0)), 1):
        query_imgs, query_labels, support_imgs, support_labels = batch
        print(query_imgs.size(), query_labels.size())
        print(support_imgs.size(), support_labels.size())
        print(support_labels)

    end_time = time.time()
    print('Cost time:{}'.format(end_time - start_time))

    # Cost time: 195.35s
    # Cost time: 1498.60
