"""
# This Conv64F network was designed following the practice of the following papers:
Revisiting Local Descriptor based Image-to-Class Measure for Few-shot Learning.
Wenbin Li, Lei Wang, Jinglin Xu, Jing Huo, Yang Gao and Jiebo Luo. In CVPR 2019.
https://github.com/WenbinLee/DN4/blob/master/models/network.py
"""

import torch.nn as nn
import functools
from engine.configs import Embeddings
from ..net_tools import get_norm_layer

@Embeddings.register('conv64f')
class Conv64F(nn.Module):
    def __init__(self, norm='batch'):
        super(Conv64F, self).__init__()
        norm_layer = get_norm_layer(norm)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.features = nn.Sequential(  # 3*84*84
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*42*42

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*21*21

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),  # 64*21*21

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),  # 64*21*21

            # TODO:
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),  # 64*21*21

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            # nn.LeakyReLU(0.2, True),  # 64*21*21
        )

        self.classifer = nn.Linear(64, 64)

    def forward(self, x):
        # return self.features(x)
        # TODO:
        # https://discuss.pytorch.org/t/model-weights-not-being-updated/1842/5
        x = self.features(x)
        # B, C, H, W = x.size()
        # x = x.permute(0, 2, 3, 1)
        # x = x.contiguous().view(B*H*W, -1)
        # x = self.classifer(x)
        # x = x.view(B, H, W, C)
        # x = x.permute(0, 3, 1, 2)
        # # print(x.size())
        return x


if __name__ == '__main__':
    import torch
    x = torch.Tensor(1, 3, 84, 84)
    net = Conv64F()
    y = net(x)
    print(y.size())  # 1x64x21x21
