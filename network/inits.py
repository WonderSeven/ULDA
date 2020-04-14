import os
import shutil
from PIL import ImageFile

import torch.nn as nn
from torch.optim import SGD
from torch.optim.adam import Adam
from torch.optim import lr_scheduler
from torchvision import transforms
from compare.autoaugment import ImageNetPolicy

from datasets.few_shot_dataset import FewShotDataSet
from datasets.few_shot_dataloader import FewShotDataloader
from torch.utils.data import DataLoader as BaseDataloader
from compare import Rotate_dataloader as SSL_dataloader
from datasets.base_dataset import BaseDataSet
from engine.logger import *
from engine import configs

from preprocess.tools import format_time

ImageFile.LOAD_TRUNCATED_IMAGES = True
sys.dont_write_bytecode = True


def get_function(registry, name):
    if name in registry:
        return registry[name]
    else:
        raise Exception("{} does not support [{}], valid keys : {}".format(registry, name, list(registry.keys())))


def get_embedding_network(cfg):
    name = cfg.model.embedding.lower()
    # assert name in ['conv64f', 'conv64f1600', 'global_conv64f1600', 'resnet12', 'resnet18', 'resnet256f']
    args = {}
    func = get_function(configs.Embeddings, name)
    return func(**args)


def get_classifier_head(cfg):
    name = cfg.model.classifier.lower()
    # assert name in ['protonet', 'relationnet', 'covamnet', 'dn4', 'dn4_batch', 'ridge', 'r2d2', 'svm_he', 'svm_cs',
    #                 'svm_ww']
    kwargs = eval(cfg.model.kwargs)
    if not isinstance(kwargs, dict):
        raise Exception("scheduler args should be string of dict, e.g. '{k1:v1}'")
    args = {}
    # if name in ['protonet', 'relationnet', 'covamnet', 'dn4', 'ridge', 'r2d2', 'svm_he', 'svm_cs', 'svm_ww']:
    func = get_function(configs.Classifers, name)

    try:
        if issubclass(func, nn.Module):
            return func(**args)
        print('Classifier is a nn.Module object')
    except:
        print('Classifier is a func')
    return func


def get_classifier(cfg):
    from network.Classifiers import ClassificationHead
    return ClassificationHead.ClassificationHead(cfg)


def get_solver(cfg, emb_net, cla_net):
    opt_name = cfg.solver.optimizer.lower()
    lr = cfg.solver.base_lr
    if opt_name == 'adam':
        return Adam([{'params': emb_net.parameters()},
                     {'params': cla_net.parameters()}],
                    lr=lr, betas=(0.5, 0.9))
    elif opt_name == 'sgd':
        wd = float(cfg.solver.weight_decay)
        momentum = cfg.solver.momentum
        return SGD([{'params': emb_net.parameters()},
                    {'params': cla_net.parameters()}],
                   lr=lr, weight_decay=wd, momentum=momentum, nesterov=True)
    else:
        raise Exception("Not support opt : {}".format(opt_name))


def get_FSL_scheduler(cfg, opt):
    scheduler_name = cfg.scheduler.name.lower()

    kwargs = eval(cfg.scheduler.kwargs)
    if not isinstance(kwargs, dict):
        raise Exception("scheduler args should be string of dict, e.g. '{k1:v1}'")
    if scheduler_name == 'step':
        return lr_scheduler.StepLR(opt, **kwargs)
    elif scheduler_name == 'lambda':
        lambda_epoch = lambda e: 1.0 if e < 20 else (0.06 if e < 40 else 0.012 if e < 50 else (0.0024))
        return lr_scheduler.LambdaLR(opt, lr_lambda=lambda_epoch, last_epoch=-1)
    else:
        raise NotImplementedError('No scheduler:{}'.format(scheduler_name))


def get_loss_func(cfg):
    name = cfg.solver.criterion
    if name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif name == 'mse':
        return nn.MSELoss()
    else:
        raise ValueError('criterion should be cross_entropy')


def get_transforms(cfg, mode):
    if not cfg.trainer.no_augment and mode == 'train':
        """
        print('Do data augmentation')
        transform = transforms.Compose([
            transforms.Resize((cfg.dataloader.image_size, cfg.dataloader.image_size)),
            transforms.RandomCrop(cfg.dataloader.image_size, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            # Cutout(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # """
        print('Do AutoAugmentation')
        transform = transforms.Compose([
            transforms.Resize((cfg.dataloader.image_size, cfg.dataloader.image_size)),
            transforms.Resize((256, 256)),
            ImageNetPolicy(),
            transforms.Resize((cfg.dataloader.image_size, cfg.dataloader.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((cfg.dataloader.image_size, cfg.dataloader.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    return transform


def get_FSL_dataset(cfg, mode):
    assert mode in ['train', 'val', 'test']
    transform = get_transforms(cfg, mode)
    dataset = FewShotDataSet(cfg, transform, mode)
    return dataset


def get_FSL_dataloader(cfg, modes):
    shuffle = cfg.dataloader.shuffle
    num_workers = cfg.dataloader.num_workers
    train_batch_size = cfg.dataloader.train_batch_size
    test_batch_size = cfg.dataloader.test_batch_size
    train_epoch_size = cfg.datasets.episode_train_num
    val_epoch_size = cfg.datasets.episode_val_num
    test_epoch_size = cfg.datasets.episode_test_num

    loaders = []
    for mode in modes:
        dataset = get_FSL_dataset(cfg, mode)
        if mode == 'train':
            # loader = FewShotDataloader(dataset, batch_size=train_batch_size, epoch_size=train_epoch_size, shuffle=shuffle,
            #                     num_workers=num_workers)
            # TODO: SSL dataloader: include UMTRA ICLR 2019.
            loader = SSL_dataloader.FewShotDataloader(dataset, batch_size=train_batch_size, epoch_size=train_epoch_size,
                                                      shuffle=shuffle, num_workers=num_workers)
        elif mode == 'val':
            loader = FewShotDataloader(dataset, batch_size=test_batch_size, epoch_size=val_epoch_size, shuffle=False,
                                       num_workers=num_workers)
        elif mode == 'test':
            loader = FewShotDataloader(dataset, batch_size=test_batch_size, epoch_size=test_epoch_size, shuffle=False,
                                       num_workers=num_workers)
        else:
            raise ValueError('Mode ought to be in [train, val, test]')
        loaders.append(loader)
    return loaders


def get_normal_dataset(cfg, mode):
    assert mode in ['train', 'val', 'test']
    transform = get_transforms(cfg, mode)
    dataset = BaseDataSet(cfg, transform, mode)
    return dataset


def get_normal_dataloader(cfg, modes):
    shuffle = cfg.dataloader.shuffle
    num_workers = cfg.dataloader.num_workers
    train_batch_size = 128
    test_batch_size = 128
    loaders = []
    for mode in modes:
        dataset = get_normal_dataset(cfg, mode)
        if mode == 'train':
            loader = BaseDataloader(dataset, batch_size=train_batch_size, shuffle=shuffle, num_workers=num_workers)

        elif mode in ['val', 'test']:
            loader = BaseDataloader(dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
        else:
            raise ValueError('Mode ought to be in [train, val, test]')
        loaders.append(loader)
    return loaders


def get_logger(cfg, name=None):
    logger = create_logger(cfg.datasets.name)
    if 'file' in cfg.trainer.loggers:
        cur_time = format_time()
        if name is None:
            log_name = 'DN4_{}_{}_{}_{}.txt'.format(cfg.datasets.name, cfg.model.embedding, cfg.trainer.stage, cur_time)
        else:
            log_name = '{}_{}_{}_{}_{}.txt'.format(name, cfg.datasets.name, cfg.model.embedding, cfg.trainer.stage,
                                                   cur_time)

        log_path = os.path.join(cfg.trainer.output_dir, log_name)
        if os.path.exists(log_path):
            os.remove(log_path)

        # save config
        shutil.copy(cfg.config, log_path)
        # save logger
        add_filehandler(logger, log_path)
    return logger

# ----------------------------------------------------------------------------
# Components for self-supervised task
# ----------------------------.__init__()------------------------------------------------

def get_ssl_classifier(feature_dim=1600, num_classes=4):
    from network.Classifiers import Linear
    # return Linear.LinearHead(feature_dim, num_classes)
    return Linear.LinearHead_V1(feature_dim, num_classes)

def get_ssl_solver(cfg, emb_net, cla_net, ssl_net):
    opt_name = cfg.solver.optimizer.lower()
    lr = cfg.solver.base_lr
    if opt_name == 'adam':
        return Adam([{'params': emb_net.parameters()},
                     {'params': cla_net.parameters()},
                     {'params': ssl_net.parameters()}],
                    lr=lr, betas=(0.5, 0.9))
    elif opt_name == 'sgd':
        wd = float(cfg.solver.weight_decay)
        momentum = cfg.solver.momentum
        return SGD([{'params': emb_net.parameters()},
                    {'params': cla_net.parameters()},
                    {'params': ssl_net.parameters()}],
                   lr=lr, weight_decay=wd, momentum=momentum, nesterov=True)
    else:
        raise Exception("Not support opt : {}".format(opt_name))


