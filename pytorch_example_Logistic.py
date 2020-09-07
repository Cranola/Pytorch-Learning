import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optimizer


class LR(nn.Module):

    def __init__(self):
        super(LR, self).__init__()
        self.fc = nn.Linear(24, 2)

    def forward(self, x):
        out = self.fc(x)
        out = F.sigmoid(out)
        return out