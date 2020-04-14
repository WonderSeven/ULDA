
# Linear classifier for self-supervised learning task
import sys
import torch
import torch.nn as nn
from engine.configs import Classifers

sys.dont_write_bytecode = True

@Classifers.register('linear')
class LinearHead(nn.Module):
    def __init__(self, feature_dim=1600, num_classes=4):
        super(LinearHead, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, query):
        tasks_per_batch = query.size(0)
        n_query = query.size(1)

        if query.dim() != 2:
            # have not been flattened
            query = query.reshape(tasks_per_batch * n_query, -1)

        d = query.size(-1)
        assert (self.feature_dim == d)

        output = self.fc(query)
        return output

@Classifers.register('linear_v1')
class LinearHead_V1(nn.Module):
    def __init__(self, feature_dim=64, num_classes=4):
        super(LinearHead_V1, self).__init__()
        self.feature_dim = feature_dim
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(64, affine=True),
            # nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, True),
            # nn.Linear(feature_dim, num_classes)
            nn.Dropout2d(),
        )
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, query):
        tasks_per_batch = query.size(0)
        n_query = query.size(1)

        if query.dim() == 5:
            # have not been flattened
            query = query.reshape([tasks_per_batch * n_query] + list(query.shape[-3:]))
        assert query.size(1) == self.feature_dim

        output = self.classifier(query)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


if __name__ == '__main__':
    # import numpy as np
    # x = torch.tensor(np.arange(10)).reshape((5, 2))
    # print(x)
    # x = x.reshape((2, 5))
    # print(x)
    # classifier = LinearHead(5, 10)
    x = torch.rand((4, 64, 5, 5)).float()
    net = LinearHead_V1(feature_dim=64)
    y = net(x)
    print(y.size())
