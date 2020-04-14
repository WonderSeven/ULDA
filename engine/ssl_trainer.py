# -*- coding:utf-8 -*-
"""
Created by 'tiexin'
"""
import os
import random
import logging
import numpy as np
from pathlib import Path
from PIL import ImageFile

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import network.inits as inits
from engine.checkpointer import Checkpointer

# from engine import baseline as bl
from engine import ssl_baseline as bl

import sys
sys.dont_write_bytecode = True

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Trainer(object):
    def __init__(self, cfg):
        print('Constructing components...')

        # basic settings
        self.cfg = cfg
        self.gpu_ids = str(cfg.gpu_ids)
        self.eps = cfg.trainer.eps
        self.epochs = cfg.trainer.epochs
        self.stage  = cfg.trainer.stage
        self.print_freq = cfg.trainer.print_freq
        self.save_freq  = cfg.trainer.save_freq
        self.output_path = Path(cfg.trainer.output_dir)
        self.output_path.mkdir(exist_ok=True)

        # seed and stage
        seed = cfg.trainer.seed
        self.set_seed(seed)

        # To cuda
        print('GPUs id:{}'.format(self.gpu_ids))
        # delete next line will cause error(output size missing)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_ids
        self.use_cuda = torch.cuda.is_available()
        cudnn.benchmark = True

        # components
        self.cla_net = inits.get_classifier(cfg)
        self.emb_net = inits.get_embedding_network(cfg)
        self.ssl_net = inits.get_ssl_classifier(feature_dim=64, num_classes=4)  # 28224
        self.opt = inits.get_ssl_solver(cfg, self.emb_net, self.cla_net, self.ssl_net)
        self.loss_func = inits.get_loss_func(cfg)
        self.train_loader, self.val_loader, self.test_loader = inits.get_FSL_dataloader(cfg, ['train', 'val', 'test'])

        self.scheduler = inits.get_FSL_scheduler(cfg, self.opt)

        # multi gpu
        if len(self.gpu_ids.split(',')) > 1:
            self.emb_net = nn.DataParallel(self.emb_net)
            self.cla_net = nn.DataParallel(self.cla_net)
            self.ssl_net = nn.DataParallel(self.ssl_net)
            print('GPUs:', torch.cuda.device_count())
            cudnn.benchmark = True
            print('Using CUDA...')
        if self.use_cuda:
            self.emb_net.cuda()
            self.cla_net.cuda()
            self.ssl_net.cuda()
            self.loss_func.cuda()

        self.test_epoch = cfg.trainer.test_epoch
        self.repeat_num = 5  # for test
        self.start_epoch = 0
        self.best_acc = 0

        # log and checkpoint
        self.checkpointer = Checkpointer(self.output_path, self.emb_net, self.cla_net, self.opt, self.scheduler)

        self.logger = inits.get_logger(cfg, cfg.model.classifier)
        self.logger.setLevel(logging.INFO)
        self.logger.info('')

        self.set_training_stage(self.stage)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def set_training_stage(self, stage):
        stage = stage.strip().lower()
        if stage == 'train':
            self.stage = 2
        elif stage == 'val' or stage == 'test':
            self.stage = 1
            # Switch to test mode
            # self.emb_net.eval()
            self.cla_net.eval()
            self.ssl_net.eval()
            self.checkpointer.load_model(self.get_load_name(self.test_epoch))

        elif stage == 'continue':
            self.stage = 2
            # name needs to be specialized
            start_model = self.get_load_name(self.start_epoch)
            self.start_epoch = self.checkpointer.load_model(start_model)

    def get_load_name(self, epoch=-1):
        if epoch == -1:
            model_name = 'best'
        elif epoch == -2:
            model_name = None
        else:
            model_name = str(epoch)
        return model_name

    def _train_net(self, epoch):
        opt = {
            'way_num': self.cfg.datasets.way_num,
            'shot_num': self.cfg.datasets.shot_num,
            'eps': self.cfg.trainer.eps}
        loss, top1, top3 = bl.train(opt, epoch, self.emb_net, self.cla_net, self.ssl_net, self.train_loader, self.opt,
                                    self.loss_func, self.use_cuda, self.print_freq)
        return loss, top1, top3
        pass

    def _val_net(self, dataloader):
        opt = {
            'way_num': self.cfg.datasets.way_num,
            'shot_num': self.cfg.datasets.shot_num}
        loss, top1, top3, accuracies = bl.test(opt, self.emb_net, self.cla_net, dataloader, self.loss_func, self.use_cuda, self.print_freq)
        return loss, top1, top3, accuracies
        pass

    def train(self):
        if self.stage >= 2:
            for epoch_item in range(self.start_epoch, self.epochs):
                print('===================================== Epoch %d =====================================' % epoch_item)
                # Fix the parameters of Batch Normalization after 10000 episodes (1 epoch)
                if epoch_item < 1:
                    self.emb_net.train()
                    self.cla_net.train()
                    self.ssl_net.train()
                else:
                    # self.emb_net.eval()
                    self.cla_net.eval()
                    self.ssl_net.eval()

                train_loss, train_top1, train_top3 = self._train_net(epoch_item)
                val_loss, val_top1, val_top3, _ = self._val_net(self.val_loader)
                test_loss, test_top1, test_top3, _ = self._val_net(self.test_loader)
                if val_top1.avg.item() > self.best_acc:
                    self.best_acc = val_top1.avg.item()
                    update_info = {'arch': self.cfg.model.embedding, 'best_prec1': self.best_acc}
                    self.checkpointer.save_model('best', epoch_item, **update_info)
                if epoch_item % self.save_freq == 0:
                    update_info = {'arch': self.cfg.model.embedding, 'best_prec1': self.best_acc}
                    self.checkpointer.save_model(str(epoch_item), epoch_item, **update_info)
                self.logger.info('Epoch:{} || Train {}, Train {}, train {}'.format(epoch_item, train_top1, train_top3, train_loss))
                self.logger.info('Epoch:{} || Val   {}, Val   {}, val   {}'.format(epoch_item, val_top1, val_top3, val_loss))
                self.logger.info('Epoch:{} || Test  {}, Test  {}, Test  {} || Best Acc@1:{}'.format(epoch_item, test_top1,
                                                                                                            test_top3, test_loss, self.best_acc))

                self.scheduler.step(epoch_item)

        elif self.stage == 1:
            total_accuracy = 0.0
            total_h = np.zeros(self.repeat_num)
            total_accuracy_vector = []
            for epoch in range(self.repeat_num):
                test_loss, test_top1, test_top3, test_accuracies = self._val_net(self.test_loader)
                test_accuracy, h = bl.mean_confidence_interval(test_accuracies)
                self.logger.info('Epoch:{} || Test {}, Test {}, Test {} || Test accuracy:{}, h:{}'.format(
                    epoch, test_top1, test_top3, test_loss, test_accuracy, h))
                total_accuracy += test_accuracy
                total_accuracy_vector.extend(test_accuracies)
                total_h[epoch] = h
            aver_accuracy, _ = bl.mean_confidence_interval(total_accuracy_vector)
            self.logger.info("Aver_accuracy:{}, Aver_h:{}".format(aver_accuracy, total_h.mean()))
        else:
            raise ValueError('Stage is wrong')
