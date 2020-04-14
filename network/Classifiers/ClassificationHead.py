import sys

import torch
import torch.nn as nn
from network.inits import get_classifier_head

sys.dont_write_bytecode = True

class ClassificationHead(nn.Module):
    def __init__(self, cfg, enable_scale=True):
        super(ClassificationHead, self).__init__()
        self.head = get_classifier_head(cfg)
        self.enable_scale = enable_scale
        self.scale = nn.Parameter(torch.FloatTensor([1.0]))

    def forward(self, query, support, support_labels, n_way, n_shot, **kwargs):
        if self.enable_scale:
            return self.scale * self.head(query, support, support_labels, n_way, n_shot, **kwargs)
        else:
            return self.head(query, support, support_labels, n_way, n_shot, **kwargs)
