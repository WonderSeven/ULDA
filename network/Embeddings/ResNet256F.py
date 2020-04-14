"""
# This Conv64F network was designed following the practice of the following papers:
Revisiting Local Descriptor based Image-to-Class Measure for Few-shot Learning.
Wenbin Li, Lei Wang, Jinglin Xu, Jing Huo, Yang Gao and Jiebo Luo. In CVPR 2019.
https://github.com/WenbinLee/DN4/blob/master/models/network.py
"""

import math
import functools
import torch.nn as nn
from ..net_tools import get_norm_layer
from engine.configs import Embeddings

class ResBlock(nn.Module):
    def __init__(self, nFin, nFout):
        super(ResBlock, self).__init__()

        self.conv_block = nn.Sequential()
        self.conv_block.add_module('BNorm1', nn.BatchNorm2d(nFin))
        self.conv_block.add_module('LRelu1', nn.LeakyReLU(0.2))
        self.conv_block.add_module('ConvL1', nn.Conv2d(nFin, nFout, kernel_size=3, padding=1, bias=False))
        self.conv_block.add_module('BNorm2', nn.BatchNorm2d(nFout))
        self.conv_block.add_module('LRelu2', nn.LeakyReLU(0.2))
        self.conv_block.add_module('ConvL2', nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False))
        self.conv_block.add_module('BNorm3', nn.BatchNorm2d(nFout))
        self.conv_block.add_module('LRelu3', nn.LeakyReLU(0.2))
        self.conv_block.add_module('ConvL3', nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False))

        self.skip_layer = nn.Conv2d(nFin, nFout, kernel_size=1, stride=1)

    def forward(self, x):
        return self.skip_layer(x) + self.conv_block(x)

@Embeddings.register('resnet256f')
class ResNet256F(nn.Module):
    def __init__(self, in_planes=3, userelu=False, dropout=0.5, norm='batch'):
        super(ResNet256F, self).__init__()
        self.in_planes = in_planes
        self.userelu = userelu
        self.dropout = dropout
        self.out_planes = [64, 96, 128, 256]
        self.num_stages = 4

        norm_layer = get_norm_layer(norm_type=norm)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]

        assert (type(self.out_planes) == list)
        assert (len(self.out_planes) == self.num_stages)
        num_planes = [self.out_planes[0], ] + self.out_planes

        self.features = nn.Sequential()
        self.features.add_module('ConvL0', nn.Conv2d(self.in_planes, num_planes[0], kernel_size=3, padding=1))

        for i in range(self.num_stages):
            self.features.add_module('ResBlock' + str(i), ResBlock(num_planes[i], num_planes[i + 1]))
            if i < self.num_stages - 2:
                self.features.add_module('MaxPool' + str(i), nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.features.add_module('ReluF1', nn.LeakyReLU(0.2, True))  # get Batch*256*21*21

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.features(x)


if __name__ == '__main__':
    import torch
    x = torch.Tensor(1, 3, 84, 84)
    net = ResNet256F()
    y = net(x)
    print(y.size())  # 1x256x21x21
