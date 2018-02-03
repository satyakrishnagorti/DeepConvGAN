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
from PIL import Image
import imageio

from generator import Generator
from discriminator import Discriminator

image_size = 64
batch_size = 32
learning_rate = 0.02
num_epochs = 5 
save_dir = './results'
z_dim = 100
betas = (0.5, 0.999)
num_filters = [1024, 512, 256, 128]


transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

mnist_data = datasets.MNIST(root='./data',
                         train=False,
                         transform=transform,
                         download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_data,
                                          batch_size=batch_size,
                                          shuffle=True)


# Plot losses
def plot_loss(d_losses, g_losses, num_epoch, save=False, save_dir='results/', show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epochs)
    ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses))*1.1)
    plt.xlabel('Epoch {0}'.format(num_epoch + 1))
    plt.ylabel('Loss values')
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.legend()

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'MNIST_DCGAN_losses_epoch_{:d}'.format(num_epoch + 1) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


def plot_result(generator, noise, num_epoch, save=False, save_dir='results/', show=False, fig_size=(5, 5)):
    generator.eval()

    noise = Variable(noise.cuda())
    gen_image = generator(noise)
    gen_image = denorm(gen_image)

    generator.train()

    n_rows = np.sqrt(noise.size()[0]).astype(np.int32)
    n_cols = np.sqrt(noise.size()[0]).astype(np.int32)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    for ax, img in zip(axes.flatten(), gen_image):
        ax.axis('off')
        ax.set_adjustable('box-forced')
        ax.imshow(img.data.view(image_size, image_size).numpy(), cmap='gray', aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    title = 'Epoch {0}'.format(num_epoch+1)
    fig.text(0.5, 0.04, title, ha='center')

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'MNIST_DCGAN_epoch_{:d}'.format(num_epoch+1) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


G = Generator(z_dim, 1, num_filters)
D = Discriminator()

# binary cross entropy loss
criterion = nn.BCELoss()
G_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=betas)
D_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=betas)

D_avg_losses = []
G_avg_losses = []


for epoch in range(num_epochs):
    D_losses = []
    G_losses = []
    print("Epoch:", epoch)
    for i, data in enumerate(data_loader,0):
        # if(i%100==0): print('batch iteration:',i)
        images, _ = data

        mini_batch = images.size()[0]
        y_real = Variable(torch.ones(mini_batch))
        y_fake = Variable(torch.zeros(mini_batch))
        X = Variable(images)

        # print(X.size())

        temp = X[0].data.numpy().squeeze()

        # discriminator training
        # vector of size batch_size
        D_real_pred = D(X).squeeze()
        D_real_loss = criterion(D_real_pred, y_real)

        z = torch.randn(mini_batch, z_dim).view(-1, z_dim, 1, 1)
        z = Variable(z)

        gen_image = G(z)

        D_fake_pred = D(gen_image).squeeze()
        D_fake_loss = criterion(D_fake_pred, y_fake)

        D_loss = D_real_loss + D_fake_loss
        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()


        # generator training
        z = torch.randn(mini_batch, z_dim).view(-1, z_dim, 1, 1)
        z = Variable(z)

        gen_image = G(z)

        D_fake_pred = D(gen_image).squeeze()
        G_loss = criterion(D_fake_pred, y_real)
        D.zero_grad()
        G.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        D_losses.append(D_loss.data[0])


        if (i%10 == 0):
            print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f' % (epoch+1, num_epochs, i+1, len(data_loader)), D_loss.data[0], G_loss.data[0]))
    
    D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
    G_avg_loss = torch.mean(torch.FloatTensor(G_losses))

    D_avg_losses.append(D_avg_loss)
    G_avg_losses.append(G_avg_loss)

    plot_loss(D_avg_losses, G_avg_losses, epoch, save=True)

    plot_result(G, fixed_noise, epoch, save=True, fig_size=(5, 5))


# Make gif
loss_plots = []
gen_image_plots = []
for epoch in range(num_epochs):
    # plot for generating gif
    save_fn1 = save_dir + 'MNIST_DCGAN_losses_epoch_{:d}'.format(epoch + 1) + '.png'
    loss_plots.append(imageio.imread(save_fn1))

    save_fn2 = save_dir + 'MNIST_DCGAN_epoch_{:d}'.format(epoch + 1) + '.png'
    gen_image_plots.append(imageio.imread(save_fn2))

imageio.mimsave(save_dir + 'MNIST_DCGAN_losses_epochs_{:d}'.format(num_epochs) + '.gif', loss_plots, fps=5)
imageio.mimsave(save_dir + 'MNIST_DCGAN_epochs_{:d}'.format(num_epochs) + '.gif', gen_image_plots, fps=5)