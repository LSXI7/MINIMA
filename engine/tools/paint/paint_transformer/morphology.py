import torch
import torch.nn as nn
import torch.nn.functional as F


class Erosion2d(nn.Module):

    def __init__(self, m=1):
        super(Erosion2d, self).__init__()
        self.m = m
        self.pad = [m, m, m, m]
        self.unfold = nn.Unfold(2 * m + 1, padding=0, stride=1)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        x_pad = F.pad(x, pad=self.pad, mode='constant', value=1e9)
        channel = self.unfold(x_pad).view(batch_size, c, -1, h, w)
        result = torch.min(channel, dim=2)[0]
        return result


def erosion(x, m=1):
    b, c, h, w = x.shape
    x_pad = F.pad(x, pad=[m, m, m, m], mode='constant', value=1e9)
    channel = nn.functional.unfold(x_pad, 2 * m + 1, padding=0, stride=1).view(b, c, -1, h, w)
    result = torch.min(channel, dim=2)[0]
    return result


class Dilation2d(nn.Module):

    def __init__(self, m=1):
        super(Dilation2d, self).__init__()
        self.m = m
        self.pad = [m, m, m, m]
        self.unfold = nn.Unfold(2 * m + 1, padding=0, stride=1)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        x_pad = F.pad(x, pad=self.pad, mode='constant', value=-1e9)
        channel = self.unfold(x_pad).view(batch_size, c, -1, h, w)
        result = torch.max(channel, dim=2)[0]
        return result


def dilation(x, m=1):
    b, c, h, w = x.shape
    x_pad = F.pad(x, pad=[m, m, m, m], mode='constant', value=-1e9)
    channel = nn.functional.unfold(x_pad, 2 * m + 1, padding=0, stride=1).view(b, c, -1, h, w)
    result = torch.max(channel, dim=2)[0]
    return result
