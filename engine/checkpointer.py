import os, sys, pdb
from pathlib import Path

import torch
import torch.nn as nn
sys.dont_write_bytecode = True


def load_model_directly(net, path, cuda=None):
    checkpoint = torch.load(path)
    pretrained_dict = checkpoint['model']

    if not isinstance(net, nn.DataParallel) and 'module.' in list(pretrained_dict.keys())[0]:
        pretrained_dict = remove_modules_for_DataParallel(pretrained_dict)

    if isinstance(net, nn.DataParallel) and 'module.' not in list(pretrained_dict.keys())[0]:
        pretrained_dict = add_modules_for_DataParallel(pretrained_dict)

    net.load_state_dict(pretrained_dict)
    print("Loaded model from {}".format(path))
    if cuda is not None:
        net = net.cuda(cuda)
    return net


def remove_modules_for_DataParallel(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' == k[:7]:
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        else:
            new_state_dict = state_dict
            break
    return new_state_dict


def add_modules_for_DataParallel(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        name = 'module.' + k
        new_state_dict[name] = v
    return new_state_dict


class Checkpointer(object):
    def __init__(self, output_path, emb_net, cla_net, optimizer, scheduler):
        self.output_path = Path(output_path)
        if not self.output_path.exists():
            self.output_path.mkdir(exist_ok=True)
        self.save_path = self.output_path / 'ckpt'
        if not self.save_path.exists():
            self.save_path.mkdir(exist_ok=True)
        self.emb_net = emb_net
        self.cla_net = cla_net
        self.optimizer = optimizer
        self.scheduler = scheduler

    def save_model(self, name, epoch, **kwargs):
        # print("Saved model to " + path)
        data = {
            'epoch_index': epoch,
            'emb_net'    : self.emb_net.state_dict(),
            'cla_net'    : self.cla_net.state_dict(),
            'optimizer'  : self.optimizer.state_dict(),
            'scheduler'  : self.scheduler.state_dict(),
        }
        data.update(kwargs)
        save_file = self.save_path / ('{}.pth.tar'.format(name))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load_model(self, name=None):
        if name is not None:
            # if specified name
            path = self.save_path / ('{}.pth.tar'.format(name))
        else:
            # if not sepcified, find the last checkpoint
            if self.has_checkpoint():
                # override argument with existing checkpoint
                path = self.get_checkpoint_file()
            else:
                path = -1
            if not path:
                # no checkpoint could be found
                print("No checkpoint found. Initializing model from scratch")
                return -1
        if Path(str(path)).is_file():
            return self.load_model_from_path(path)
        else:
            print("No checkpoint found. Initializing model from scratch")
            return -1

    def load_model_from_path(self, path):
        checkpoint = torch.load(str(path))
        pretrained_emb_dict = checkpoint['emb_net']
        pretrained_cla_dict = checkpoint['cla_net']

        # '''
        if not isinstance(self.emb_net, nn.DataParallel) and 'module.' in list(pretrained_emb_dict.keys())[0]:
            pretrained_emb_dict = remove_modules_for_DataParallel(pretrained_emb_dict)

        if isinstance(self.emb_net, nn.DataParallel) and 'module.' not in list(pretrained_emb_dict.keys())[0]:
            pretrained_emb_dict = add_modules_for_DataParallel(pretrained_emb_dict)

        if not isinstance(self.cla_net, nn.DataParallel) and 'module.' in list(pretrained_cla_dict.keys())[0]:
            pretrained_cla_dict = remove_modules_for_DataParallel(pretrained_cla_dict)

        if isinstance(self.cla_net, nn.DataParallel) and 'module.' not in list(pretrained_cla_dict.keys())[0]:
            pretrained_cla_dict = add_modules_for_DataParallel(pretrained_cla_dict)
        # '''
        # model_dict = self.net.state_dict()
        # # 1. filter out unnecessary keys
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        self.emb_net.load_state_dict(pretrained_emb_dict)
        self.cla_net.load_state_dict(pretrained_cla_dict)

        if 'optimizer' in checkpoint.keys():
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print('The model params have been changed.')

        if 'epoch_index' in checkpoint.keys():
            epoch = checkpoint['epoch_index']

        print("Loaded model from {}, lr={}, epoch={}".format(path, self.optimizer.param_groups[0]['lr'], epoch))
        return epoch

    def has_checkpoint(self):
        save_file = os.path.join(self.output_path, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.output_path, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.output_path, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(str(last_filename))