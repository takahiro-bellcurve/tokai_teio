import argparse
import os
import numpy as np
import pandas as pd
import math
import itertools
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets


# from mnist_img_dataloader import load_mnist_data
from src.lib.image_dataset.image_dataset import ImageDataset

#######################################################
# hyperparameter setting
#######################################################

parser = argparse.ArgumentParser("Basic aae model by hyu")
parser.add_argument("--n_epochs", type=int, default=500,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64,
                    help="size of the batches")

parser.add_argument("--lr", type=float, default=0.0002,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")

parser.add_argument("--latent_dim", type=int, default=10,
                    help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=32,
                    help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1,
                    help="number of image channels")

parser.add_argument("--img_dir", type=str, default='outputs/aae_basic',
                    help="number of classes of image datasets")

args = parser.parse_args()
print(args)

DATASET_DIR = 'preprocessed_images_512'

# config cuda
cuda = torch.cuda.is_available()
img_shape = (args.channels, args.img_size, args.img_size)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


#######################################################
# Define Networks
#######################################################

class Encoder(nn.Module):
    def __init__(self, input_channels, img_size, latent_dim):
        super(Encoder, self).__init__()

        self.input_channels = input_channels
        self.img_size = img_size
        self.latent_dim = latent_dim

        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveMaxPool2d((4, 4))  # 出力サイズを4x4に適応させる
        )

        # フラット化された特徴のサイズを計算
        self.flattened_size = 32 * 4 * 4

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, self.latent_dim)
        )

    def forward(self, img):
        x = self.features(img)
        x = x.view(x.size(0), -1)  # 特徴をフラット化
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, output_channels, img_size, latent_dim):
        super(Decoder, self).__init__()

        self.output_channels = output_channels
        self.img_size = img_size
        self.latent_dim = latent_dim

        # 潜在ベクトルから初期特徴マップサイズへの変換
        self.init_size = img_size // 4  # アップサンプリングのための初期サイズ
        self.l1 = nn.Sequential(
            nn.Linear(self.latent_dim, 32 * self.init_size ** 2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, self.output_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 32, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


#######################################################
# Preparation part
#######################################################

# tensorboard
writer = SummaryWriter()

# data
transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    # transforms.Grayscale(),
    transforms.ToTensor(),
])
# train_labeled_loader = load_mnist_data('./dataset/mnist/')[1]
dataset = ImageDataset(
    directory=f'data/{DATASET_DIR}', transform=transform)
train_labeled_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True)

# define model
# 1) generator
encoder = Encoder(args.channels, args.img_size, args.latent_dim)
decoder = Decoder(args.channels, args.img_size, args.latent_dim)
# 2) discriminator
discriminator = Discriminator(args.latent_dim)

# loss
adversarial_loss = nn.BCELoss()
reconstruction_loss = nn.MSELoss()

# optimizer
optimizer_G = torch.optim.Adam(itertools.chain(encoder.parameters(
), decoder.parameters()), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    reconstruction_loss.cuda()


#######################################################
# Training part
#######################################################

def sample_image(n_row, epoch, img_dir):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    z = Variable(Tensor(np.random.normal(0, 1, (n_row**2, args.latent_dim))))
    generated_imgs = decoder(z)
    save_image(generated_imgs.data, os.path.join(
        img_dir, "%depoch.png" % epoch), nrow=n_row, normalize=True)


train_logs = []
# training phase
for epoch in range(args.n_epochs):
    # for i, (x, idx) in enumerate(train_labeled_loader):
    for i, x in enumerate(train_labeled_loader):

        valid = Variable(Tensor(x.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(x.shape[0], 1).fill_(0.0), requires_grad=False)

        if cuda:
            x = x.cuda()

        # 1) reconstruction + generator loss
        optimizer_G.zero_grad()
        fake_z = encoder(x)
        # print("fake_z shape: ", fake_z.shape)
        decoded_x = decoder(fake_z)
        validity_fake_z = discriminator(fake_z)
        G_loss = 0.001 * \
            adversarial_loss(validity_fake_z, valid) + 0.999 * \
            reconstruction_loss(decoded_x, x)
        G_loss.backward()
        optimizer_G.step()

        # 2) discriminator loss
        optimizer_D.zero_grad()
        real_z = Variable(Tensor(np.random.normal(
            0, 1, (x.shape[0], args.latent_dim))))
        real_loss = adversarial_loss(discriminator(real_z), valid)
        fake_loss = adversarial_loss(discriminator(fake_z.detach()), fake)
        D_loss = 0.5*(real_loss + fake_loss)
        D_loss.backward()
        optimizer_D.step()

        writer.add_scalar('G_loss', G_loss.item(), epoch)
        writer.add_scalar('D_loss', D_loss.item(), epoch)

        # save log
        log = {
            'epoch': epoch,
            'G_loss': G_loss.item(),
            'D_loss': D_loss.item()
        }
        train_logs.append(log)
    # print loss
    print(
        "[Epoch %d/%d] [G loss: %f] [D loss: %f]"
        % (epoch, args.n_epochs, G_loss.item(), D_loss.item())
    )

    sample_image(n_row=5, epoch=epoch,
                 img_dir=f"{args.img_dir}_{args.img_size}_ch{args.channels}")

now = datetime.now().strftime("%Y-%m-%d_%H:%M")
df = pd.DataFrame(train_logs)
df.to_csv(f"logs/aae_basic_train_logs_{now}.csv")
torch.save(
    encoder, f'trained_models/aae_{args.img_size}_ch{args.channels}_encoder_{now}.pth')
