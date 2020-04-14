import os
import sys
import csv
import time
import shutil
import numpy as np
import torch
from PIL import Image
from matplotlib import pylab as plt
from torchvision.transforms import functional as F

sys.dont_write_bytecode = True


def load_csv2dict(csv_path):
    class_img_dict = {}
    with open(csv_path) as csv_file:
        csv_context = csv.reader(csv_file, delimiter=',')
        for line in csv_context:
            if csv_context.line_num == 1:
                continue
            img_name, img_class = line

            if img_class in class_img_dict:
                class_img_dict[img_class].append(img_name)
            else:
                class_img_dict[img_class] = []
                class_img_dict[img_class].append(img_name)
    class_list = class_img_dict.keys()
    return class_img_dict, class_list


def read_csv(filePath):
    imgs = []
    with open(filePath, 'r') as csvFile:
        next(csvFile)
        lines = csvFile.readlines()
        for line in lines:
            context = line.split(',')
            try:
                imgs.append((context[0].strip(), context[1].strip()))
            except:
                print('Escape context:{}'.format(context))
    return imgs

def write_csv(mlist, file_path):
    out = open(file_path, 'w', newline='')
    csv_writer = csv.writer(out, dialect='excel')
    for row in mlist:
        csv_writer.writerow(row)
    out.close()

def load_class_names(file_path):
    names = {}
    with open(file_path) as f:
        for line in f:
            pieces = line.strip().split(',')
            class_id = int(pieces[1])
            names[pieces[0]] = class_id
    return names


def format_time():
    return time.strftime("%Y-%m-%d-%H:%M", time.localtime(time.time()))

# if __name__ == '__main__':
#     train_csv_path = '../datasets/miniImageNet/train.csv'
#     train_img_dict, train_class_list = load_csv2dict(train_csv_path)
#     pass

def tensor2pil(input):
    '''
    :param input: tensor(1x3x84x84)
    :return: Image.size(84x84x3)
    '''
    input = input.squeeze()
    input = renormalize(input, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return F.to_pil_image(input)

def pil2tensor(input):
    '''
    :param input:Image.size(84x84x3)
    :return: tensor(1x3x84x84)
    '''
    input = F.to_tensor(input)
    input = F.normalize(input, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return input.unsqueeze(0)

def pil2numpy(input):
    return np.array(input)

def renormalize(tensor: torch.Tensor, mean, std, inplace=False):
    '''
    recover from F.normalization
    :param tensor:
    :param mean:
    :param std:
    :param inplace:
    :return:
    '''
    if not F._is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')
    if not inplace:
        tensor = tensor.clone()
    mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    return tensor


def l2_normalize(x, dim=1):
    return x / torch.sqrt(torch.sum(x**2, dim=dim).unsqueeze(dim))


def repeat_1d_tensor(t, num_reps):
    return t.unsqueeze(1).expand(-1, num_reps)

def shuffle_channel(x):
    batch_size, channels, height, width = x.size()
    index = torch.randperm(channels).to(x.device)
    x = x[:, index, ...]
    return x
