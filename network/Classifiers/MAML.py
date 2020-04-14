"""
url:
1. https://github.com/wyharveychen/CloserLookFewShot
2.
"""
import os
import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from network.net_tools import one_hot
from engine.configs import Classifers

# @Classifers.register('maml')
class MAMLHead(nn.Module):
    def __init__(self, emb_net, cla_net,  n_way, n_shot, approx=False):
        super(MAMLHead, self).__init__()
        self.emb_net = emb_net
        self.cla_net = cla_net
        self.n_way = n_way
        self.n_shot = n_shot

        self.loss_func = nn.CrossEntropyLoss()

        self.n_task = 4
        self.task_update_num = 5
        self.approx = approx  # first order approx.
        self.train_lr = 0.01

    def forward(self, input):
        output = self.emb_net(input)
        output = output.view(output.size(0), -1)
        scores = self.cla_net(output)
        return scores

    def inner_loop(self, query_images, query_labels, support_images, support_labels):
        """
        Format the Input data
        """
        support_images = Variable(support_images)
        support_labels = Variable(support_labels)

        batch_size = support_images.size(0)
        fast_parameters = list(self.parameters())
        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()

        for task_step in range(batch_size):
            # TODO: [2019-12-11] 内层循环 batch多次?
            input_s = support_images[task_step, :].contiguous()
            input_s_label = support_labels[task_step, :].contiguous()
            scores = self.forward(input_s)  # [5, 5]
            set_loss = self.loss_func(scores, input_s_label)
            grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True)  # tuple(14, tensor) build full graph support gradient of gradient
            if self.approx:
                grad = [g.detach() for g in grad]  # do not calculate gradient of gradient if using first order approximation
            fast_parameters = []
            # '''
            for k, weight in enumerate(self.parameters()):
                # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
                if weight.fast is None:
                    weight.fast = weight - self.train_lr * grad[k]  # create weight.fast
                else:
                    weight.fast = weight.fast - self.train_lr * grad[k]  # create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast
                fast_parameters.append(weight.fast)  # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts
            # fast_parameters = [nn.Parameter(item) for item in fast_parameters]
            pass
            # '''
        # TODO: [2019-12-11] 外层循环的query image选择batch的累计loss
        for task_step in range(batch_size):
            scores = self.forward(query_images)
            loss = self.loss_func(scores, query_labels)
        return scores, loss


# def MAMLHead(query, support, support_labels, n_way, n_shot, n_task=4, task_update_num=5):
#     """
#     :param query: (8, 75, 64, 21, 21)
#     :param support: (8, 5, 64, 21, 21)
#     :param support_labels: (8, 5)
#     :param n_way: 5
#     :param n_shot: 1
#     :param neighbor_k: 3
#     :return:
#     """
#
#     tasks_per_batch = query.size(0)
#     n_support = support.size(1)
#     n_query = query.size(1)
#     query = query.reshape(tasks_per_batch, n_query, -1)
#     support = support.reshape(tasks_per_batch, n_support, -1)
#     d = query.size(2)
#     pass


if __name__ == '__main__':
    from engine.configs.parser import BaseOptions
    import network.inits as inits
    from network.Classifiers import MAML_components
    opts = BaseOptions().opts

    n_way = opts.datasets.way_num
    n_shot = opts.datasets.shot_num
    n_query = opts.datasets.query_num

    # emb_net = inits.get_embedding_network(opts)
    emb_net = MAML_components.Conv4()
    cla_net = MAML_components.Linear_fw(in_features=1600, out_features=n_way)
    cla_net.bias.data.fill_(0)

    maml = MAMLHead(emb_net, cla_net, n_way, n_shot)

    Query_images = torch.Tensor(8, 75, 3, 84, 84).float()
    Query_labels = torch.ones([8, 75]).long()
    Support_images = torch.Tensor(8, 5, 3, 84, 84).float()
    Support_labels = torch.zeros([8, 5]).long()
    # output = maml(Query_images[0])
    # print(output.size())
    #
    # label = np.repeat(range(5), 15)
    # print(label)

    scores, loss = maml.inner_loop(Query_images, Query_labels, Support_images, Support_labels)
