"""
# This Conv64F network was designed following the practice of the following papers:
Revisiting Local Descriptor based Image-to-Class Measure for Few-shot Learning.
Wenbin Li, Lei Wang, Jinglin Xu, Jing Huo, Yang Gao and Jiebo Luo. In CVPR 2019.
https://github.com/WenbinLee/DN4/blob/master/models/network.py
"""

import torch.nn as nn
import functools
from engine.configs import Embeddings
# from network.Embeddings.tools import get_norm_layer
from ..net_tools import get_norm_layer

@Embeddings.register('global_conv64f1600')
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
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*10*10

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*5*5
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(64, affine=True),
            # nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 1600, kernel_size=1),
            # nn.InstanceNorm2d(1600, affine=True),
            nn.BatchNorm2d(1600, affine=True),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        x = self.features(x)
        # TODO: Global feature
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x


if __name__ == '__main__':
    import torch
    x = torch.Tensor(6, 3, 84, 84)
    net = Conv64F()
    y = net(x)
    print(y.size())  # 1x64x5x5
