import sys
import numpy as np

import torch
from preprocess.visualize import show_Images
sys.dont_write_bytecode = True

def intra_task_mixup(x, y, alpha, use_cuda=True):
    """
    :param x:
    :param y: one-hot
    :param alpha:
    :param use_cuda:
    :return:
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[1]

    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1-lam) * x[:, index, :]
    y_a, y_b = y, y[:, index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return (lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)).mean()


def intra_task_uniform__mixup(x, y, alpha, use_cuda=False):
    """
    :param x:
    :param y: one-hot
    :param alpha:
    :param use_cuda:
    :return:
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = np.maximum(lam, 1 - lam)
    else:
        lam = 1

    batch_size = x.size()[1]

    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1-lam) * x[:, index, :]

    # tmp = x[:, index, :]
    # show_Images(x[0], 3, 'Raw x')
    # show_Images(tmp[0], 3, 'Shuffled x')
    # show_Images(mixed_x[0], 3, 'Mixed x')

    y_a, y_b = y, y[:, index]
    return mixed_x, y_a, y_b, lam


def inter_task_mixup(x, y, alpha, use_cuda=True):
    """
    如何在task之间进行 mixup 考虑分布的情况
    :param x:
    :param y:
    :param alpha:
    :param use_cuda:
    :return:
    """
    pass

def batch_task_mix(x, alpha, use_cuda=True):
    assert x.dim() == 5
    batch_size = x.size(0)
    N_imgs = x.size(1)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = np.maximum(lam, 1 - lam)
    else:
        lam = 1

    if use_cuda:
        index = torch.randperm(N_imgs).cuda()
    else:
        index = torch.randperm(N_imgs)

    mixed_x = x.clone()
    for i in range(batch_size):
        mixed_x[i] = lam * x[i] + (1 - lam) * x[i, index]
    return mixed_x


# -----------------------------------------------------------------------------
# Limeng Qiao, Yemin Shi, Jia Li, Yaowei Wang, Tiejun Huang and Yonghong Tian.
# Transductive Episodic-Wise Adaptive Metric for Few-Shot Learning, ICCV, 2019.
# -----------------------------------------------------------------------------
def TIM(x, y, alpha, use_cuda=True):
    """
    :param x:
    :param y: one-hot
    :param alpha:
    :param use_cuda:
    :return:
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        # lam += 0.5
        lam = np.maximum(lam, 1 - lam)
    else:
        lam = 1

    batch_size = x.size()[1]

    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1-lam) * x[:, index, :]
    y_a, y_b = y, y[:, index]
    return mixed_x, y_a, y_b, lam


def TIM_S(x, alpha, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam += 0.5
        # lam = np.maximum(lam, 1 - lam)
    else:
        lam = 1
    # print(lam)
    batch_size = x.size()[1]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[:, index, :]
    return mixed_x


def JS_TIM(data, n_hotels=6, length=10, use_cuda=True):
    # [task_per_batch, ]
    h, w = data.size()[-2:]
    batch_size = data.size()[1]

    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    shuffled_data = data[:, index, :]
    for n in range(n_hotels):
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        data[:, :, :, y1: y2, x1: x2] = shuffled_data[:, :, :, y1: y2, x1: x2]
    return data


if __name__ == '__main__':
    x = torch.ones((8, 5, 3, 84, 84))
    y = torch.rand(8, 5, 3, 84, 84)

    print(x.size())
    mixed_x = batch_task_mix(x, alpha=1.0, use_cuda=False)
    print(mixed_x)
