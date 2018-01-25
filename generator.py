import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as tranforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Generator(nn.Module):

    def __init__(self, input_dim, output_dim, filter_sizes): 
        super(Generator, self).__init__()

        self.modelG = nn.Sequential()
 
        for i, _ in enumerate(filter_sizes):
            if(i == 0):
                transpConv = torch.nn.ConvTranspose2d(input_dim, filter_sizes[i], kernel_size=4, stride=1, padding=0)
            else:
                transpConv = torch.nn.ConvTranspose2d(filter_sizes[i-1], filter_sizes[i], kernel_size=4, stride=2, padding=1)

            #init as mentioned in DCGAN paper
            torch.nn.init.normal(transpConv.weight, mean=0, std=0.02)
            torch.nn.init.constant(transpConv.bias, 0.0)

            batchNorm = torch.nn.BatchNorm2d(filter_sizes[i])

            self.modelG.add_module('transpConv' + str(i), transpConv)
            self.modelG.add_module('batchNorm' + str(i), batchNorm)
            self.modelG.add_module('activation' + str(i), nn.ReLU())

        finalTranspConv = torch.nn.ConvTranspose2d(filter_sizes[len(filter_sizes)-1], output_dim, kernel_size=4, stride=2, padding=1)
        torch.nn.init.normal(finalTranspConv.weight, mean=0, std=0.02)
        torch.nn.init.constant(finalTranspConv.bias, 0.0)
 
        self.modelG.add_module('output', finalTranspConv)
        self.modelG.add_module('final_act', nn.Tanh())


    def forward(self, x):
        return self.modelG(x)
