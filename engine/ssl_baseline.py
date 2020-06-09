'''
Author: Tiexin
Time: 2019-8-15
'''

from __future__ import print_function
import sys
import time
import numpy as np
import scipy as sp
import scipy.stats
import torch
import torch.nn.functional as F
from .averagemeter import AverageMeter, ProgressMeter
from network.net_tools import one_hot
from preprocess.mixup import TIM, TIM_S
from preprocess.tools import shuffle_channel
from preprocess.visualize import show_Images

sys.dont_write_bytecode = True


def train(opt, epoch, emb_net, cla_net, ssl_net, train_loader, optimizer, criterion, use_cuda=True, print_freq=1):
    way_num = opt['way_num']
    shot_num = opt['shot_num']
    eps = opt['eps']

    batch_time = AverageMeter('Time', ':6.3f')
    losses     = AverageMeter('Loss', ':.4e')
    top1       = AverageMeter('Acc@1', ':6.3f')
    top3       = AverageMeter('Acc@3', ':6.3f')
    ssl_top1   = AverageMeter('SSL_Acc1@1', ':6.3f')
    progress   = ProgressMeter(len(train_loader), batch_time, losses, top1,
                               top3, ssl_top1, prefix="Epoch: [{}]".format(epoch))

    # emb_net.train()
    # cla_net.train()
    # ssl_net.train()

    if print_freq is None:
        print_freq = len(train_loader)

    end = time.time()
    for episode_index, (batch) in enumerate(train_loader(epoch)):
        if use_cuda:
            query_images, query_targets, ssl_images, ssl_labels, support_images, support_targets = [item.cuda() for item in batch]
        else:
            query_images, query_targets, ssl_images, ssl_labels, support_images, support_targets = batch
        """
        query_images:    (8, 75, 3, 84, 84)       ==> (batch_size, query_nums * way_nums, channels, H, W)
        query_targets:   (8, 20)                  ==> (batch_size, query_nums * rotate_angles)
        query_rotation_labels: (8, 20)            ==> (batch_size, query_nums * rotate_angles)
        support_images:  (8, 5, 3, 84, 84)        ==> (batch_size, shot_nums * way_nums, channels, H, W)
        support_targets: (8, 5)                   ==> (batch_size, shot_nums * way_nums)
        """
        batch_size   = support_targets.shape[0]
        support_nums = support_targets.shape[-1]
        query_nums   = query_targets.shape[-1]
        rotate_nums  = ssl_labels.shape[-1]

        # TODO: TIM
        support_images = TIM_S(support_images, 0.8, use_cuda)
        ssl_images, q_targets_a, q_targets_b, lam = TIM(ssl_images, query_targets, alpha=0.6, use_cuda=use_cuda)
        query_images, q_targets_a, q_targets_b, lam = TIM(query_images, query_targets, alpha=0.6, use_cuda=use_cuda)

        # Show images
        from preprocess.visualize import show_Images
        # show_Images(support_images[0], 5, 'Support images')
        # show_Images(query_images[0], 5, 'Query images')
        # show_Images(ssl_images[0], 4, 'SSL images')

        # ================================ Support images ========================================
        emb_support = emb_net(support_images.reshape([-1] + list(support_images.shape[-3:])))  # [40, 64, 21, 21]
        if emb_support.dim() == 2:
            emb_support = emb_support.reshape([batch_size, support_nums, -1])  # [8, 5, 64, 21, 21]
        else:
            emb_support = emb_support.reshape([batch_size, support_nums] + list(emb_support.shape[-3:]))  # [8, 5, 64, 21, 21]

        """
        # ================================ Query images =========================================
        emb_query = emb_net(query_images.reshape([-1] + list(query_images.shape[-3:])))  # [600, 64, 21, 21]
        # TODO: 2. Shuffle the channels of query features
        # emb_query = shuffle_channel(emb_query)

        if emb_query.dim() == 2:
            emb_query = emb_query.reshape([batch_size, query_nums, -1])  # [8, 75, 64, 21, 21]
        else:
            emb_query = emb_query.reshape([batch_size, query_nums] + list(emb_query.shape[-3:]))  # [8, 75, 64, 21, 21]

        output = cla_net(emb_query, emb_support, support_targets, way_num, shot_num)  # [8, 75, 5]
        """

        # """
        # =============================== SSL images ==========================================
        emb_rotate = emb_net(ssl_images.reshape([-1] + list(ssl_images.shape[-3:])))  # [600, 64, 21, 21]
        # TODO: shuffle mean and std
        # emb_rotate = shuffle_mean_std(emb_rotate, True)

        if emb_rotate.dim() == 2:
            emb_rotate = emb_rotate.reshape([batch_size, rotate_nums, -1])  # [8, 75, 64, 21, 21]
        else:
            emb_rotate = emb_rotate.reshape([batch_size, rotate_nums] + list(emb_rotate.shape[-3:]))  # [8, 75, 64, 21, 21]

        ssl_output = ssl_net(emb_rotate)
        output = cla_net(emb_rotate, emb_support, support_targets, way_num, shot_num)  # [8, 75, 5]
        # """
        if eps > 0.:
            smoothed_one_hot = one_hot(query_targets.reshape(-1), way_num)
            smoothed_one_hot = smoothed_one_hot * (1 - eps) + (1 - smoothed_one_hot) * eps / (way_num - 1)
            log_prob = F.log_softmax(output.reshape(-1, way_num), dim=1)
            loss = -(smoothed_one_hot * log_prob).sum(dim=1)
            loss = loss.mean()
        else:
            output = output.reshape(-1, way_num)  # [8x75, 5]
            query_targets = query_targets.reshape(-1)  # [8*75]

            ssl_labels = ssl_labels.reshape(-1)
            # 1. FSL loss func
            # loss = criterion(output, query_targets)
            # 2. SSL loss func
            # loss = criterion(ssl_output, rotate_labels)
            # 3. FSL+SSL loss func
            loss = criterion(output, query_targets) + criterion(ssl_output, ssl_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc3 = accuracy(output, query_targets, topk=(1, 3))
        ssl_acc1, _ = accuracy(ssl_output, ssl_labels, topk=(1, 3))
        losses.update(loss.item(), batch_size)
        top1.update(acc1[0], batch_size)
        top3.update(acc3[0], batch_size)
        ssl_top1.update(ssl_acc1[0], batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if episode_index % print_freq == 0 and episode_index != 0:
            progress.print(episode_index)
    return losses, top1, top3


def test(opt, emb_net, cla_net, test_loader, criterion, use_cuda=True, print_freq=100):
    way_num = opt['way_num']
    shot_num = opt['shot_num']

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top3 = AverageMeter('Acc@3', ':6.2f')
    progress = ProgressMeter(len(test_loader), batch_time, losses, top1, top3,
                             prefix='Test: ')

    # emb_net.eval()
    # cla_net.eval()
    accuracies = []

    with torch.no_grad():
        end = time.time()
        for episode_index, (batch) in enumerate(test_loader(0)):
            if use_cuda:
                query_images, query_targets, support_images, support_targets = [item.cuda() for item in batch]
            else:
                query_images, query_targets, support_images, support_targets = batch

            """
            query_images:    (8, 75, 3, 84, 84)       ==> (batch_size, query_nums * way_nums, channels, H, W)
            query_targets:   (8, 75)                  ==> (batch_size, query_nums * way_nums)
            support_images:  (8, 5, 3, 84, 84)        ==> (batch_size, shot_nums * way_nums, channels, H, W)
            support_targets: (8, 5)                   ==> (batch_size, shot_nums * way_nums)
            """
            batch_size = support_targets.shape[0]
            support_nums = support_targets.shape[-1]
            query_nums = query_targets.shape[-1]

            emb_support = emb_net(support_images.reshape([-1] + list(support_images.shape[-3:])))  # [40, 64, 21, 21]
            emb_query = emb_net(query_images.reshape([-1] + list(query_images.shape[-3:])))  # [600, 64, 21, 21]

            emb_support = emb_support.reshape([batch_size, support_nums] + list(emb_support.shape[1:]))
            emb_query = emb_query.reshape([batch_size, query_nums] + list(emb_query.shape[1:]))

            output = cla_net(emb_query, emb_support, support_targets, way_num, shot_num)  # [8, 75, 5]

            output = output.reshape(-1, way_num)  # [8x75, 5]
            query_targets = query_targets.reshape(-1)  # [8*75]
            loss = criterion(output, query_targets)

            acc1, acc3 = accuracy(output, query_targets, topk=(1, 3))
            losses.update(loss.item(), query_images.size(0))
            top1.update(acc1[0], query_images.size(0))
            top3.update(acc3[0], query_images.size(0))
            accuracies.append(acc1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if episode_index % print_freq == 0 and episode_index != 0:
                progress.print(episode_index)
    return losses, top1, top3, accuracies

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy(%) over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(cfg, optimizer, epoch_num):
    """Sets the learning rate to the initial LR decayed by 0.05 every 10 epochs"""
    lr = cfg.solver.base_lr * (0.05 ** (epoch_num // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def mean_confidence_interval(data, confidence=0.95):
    a = [1.0*np.array(data[i].cpu()) for i in range(len(data))]
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h
