import argparse
import os
import numpy as np
import pandas as pd
import math
import itertools
from datetime import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from src.networks.v2 import Encoder, Decoder, Discriminator
from src.lib.create_data_loader import create_data_loader

#######################################################
# hyperparameter setting
#######################################################

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256,
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
parser.add_argument("--img_dir", type=str, default='outputs/aae_v2',
                    help="number of classes of image datasets")

args = parser.parse_args()
print(args)

MODEL_NAME = "aae_v2"
DATASET_DIR = 'preprocessed_images_512'

#######################################################
# Preparing part
#######################################################

# config cuda
if not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

# define file name
file_name = f"{MODEL_NAME}_{args.img_size}_ch{args.channels}_ldim_{args.latent_dim}_bs_{args.batch_size}_lr_{args.lr}_b1_{args.b1}_b2_{args.b2}"

# define Tensor
Tensor = torch.cuda.FloatTensor

# define tensorboard
writer = SummaryWriter()

# define data_loader
data_loader = create_data_loader(
    img_size=args.img_size, batch_size=args.batch_size, channels=args.channels)

# define model
# 1) generator
encoder = Encoder(args.channels, args.img_size, args.latent_dim)
decoder = Decoder(args.channels, args.img_size, args.latent_dim)
# 2) discriminator
discriminator = Discriminator(args.latent_dim)

# define loss function
adversarial_loss = nn.BCELoss()
reconstruction_loss = nn.MSELoss()

# define optimizer
optimizer_G = torch.optim.Adam(itertools.chain(encoder.parameters(
), decoder.parameters()), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

# move to cuda
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
for epoch in range(args.n_epochs):
    for i, x in enumerate(data_loader):

        valid = Variable(Tensor(x.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(x.shape[0], 1).fill_(0.0), requires_grad=False)
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
                 img_dir=f"{args.img_dir}_{args.img_size}_ch{args.channels}_ldim_{args.latent_dim}_bs_{args.batch_size}")

finished_at = datetime.now().strftime("%Y-%m-%d_%H:%M")
df = pd.DataFrame(train_logs)
df.to_csv(f"logs/{file_name}_{finished_at}.csv")
torch.save(encoder.state_dict(),
           f'trained_models/encoder/{file_name}_{finished_at}.pth')

torch.save(decoder.state_dict(),
           f'trained_models/decoder/{file_name}_{finished_at}.pth')
