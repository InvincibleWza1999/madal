import torch
import torch.nn as nn

class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return y