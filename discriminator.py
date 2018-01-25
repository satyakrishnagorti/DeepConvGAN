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

    def __init__(self, input_dim, output_dim, filter_sizes):
        super(Discriminator, self).__init__()

        self.modelD = torch.nn.Sequential()

        for i in range(filter_sizes):
            if(i==0):
                conv = torch.nn.Conv2d(input_dim, filter_sizes[i], kernel_size=4, stride=2, padding=1)
            else:
                conv = torch.nn.Conv2d(filter_sizes[i-1], filter_sizes[i], kernel_size=4, stride=2, padding=1)
        
            torch.nn.init.normal(conv.weight, mean=0, std=0.02)
            torch.nn.init.constant(conv.bias, 0.0)
        
            batchNorm = torch.nn.BatchNorm2d(filter_sizes[i])
            
            self.modelD.add_module('conv' + str(i), conv)
            if(i!=0): self.modelD.add_module('batchNorm' + str(i), batchNorm)
            self.modelD.add_module('act' + str(i), nn.LeakyReLU(0.2))

        
        finalConv = torch.nn.Conv2d(filter_sizes[len(filter_sizes)-1], output_dim, kernel_size=4, stride=1, padding=0)

        torch.nn.init.normal(finalConv.weight, mean=0, std=0.02)
        torch.nn.init.constant(finalConv.bias, 0.0)

        self.modelD.add_module('finalConv', conv)
        self.modelD.add_module('sigmoid', nn.Sigmoid())

    def forward(self, x):
        return self.modelD(x)
