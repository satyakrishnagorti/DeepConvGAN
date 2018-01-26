import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as tranforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.modelD = nn.Sequential(
            # nx1x64x64 -> nx128x32x32
            nn.Conv2d(1, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # nx128x32x32 ->  nx256x16x16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # nx256x16x16 -> nx512x8x8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # nx512x8x8 -> nx1024x4x4
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            # nx1024x4x4 -> nx1x1x1
            nn.Conv2d(1024, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.modelD(x)
