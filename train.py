import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from generator import Generator
from discriminator import Discriminator

image_size = 64
batch_size = 32
learning_rate = 0.02
num_epochs = 20
save_dir = './saved_models'
z_dim = 100
betas = (0.5, 0.999)
num_filters = [1024, 512, 256, 128]


transform = transforms.Compose([transforms.Scale(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

mnist_data = datasets.MNIST(root='./data',
                         train=True,
                         transform=transform,
                         download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_data,
                                          batch_size=batch_size,
                                          shuffle=True)


G = Generator(z_dim, 1, num_filters)
D = Discriminator(1, 1, num_filters[::-1])

# binary cross entropy loss
criterion = nn.BCELoss()
G_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=betas)
D_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=betas)

