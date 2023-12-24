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
import wandb

from src.networks.v0 import Encoder, Decoder, Discriminator
from src.lib.create_data_loader import create_data_loader

MODEL_NAME = "aae_v0"
DATASET_DIR = 'preprocessed_images_512'
SEND_WANDB = True

#######################################################
# Parameter setting
#######################################################
N_EPOCHS = 200
BATCH_SIZE = 512
LR = 0.0002
B1 = 0.5
B2 = 0.999
LATENT_DIM = 20
IMG_SIZE = 64
CHANNELS = 3

#######################################################
# Preparing part
#######################################################

# config cuda
if not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

# define file name
file_name = f"{MODEL_NAME}_{IMG_SIZE}_ch{CHANNELS}_ldim_{LATENT_DIM}_bs_{BATCH_SIZE}_lr_{LR}_b1_{B1}_b2_{B2}"

# define Tensor
Tensor = torch.cuda.FloatTensor

# define data_loader
data_loader = create_data_loader(
    img_size=IMG_SIZE, batch_size=BATCH_SIZE, channels=CHANNELS)

# define model
# 1) generator
encoder = Encoder(CHANNELS, IMG_SIZE, LATENT_DIM)
decoder = Decoder(CHANNELS, IMG_SIZE, LATENT_DIM)
# 2) discriminator
discriminator = Discriminator(LATENT_DIM)

# define loss function
adversarial_loss = nn.BCELoss()
reconstruction_loss = nn.MSELoss()

# define optimizer
optimizer_G = torch.optim.Adam(itertools.chain(encoder.parameters(
), decoder.parameters()), lr=LR, betas=(B1, B2))
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=LR, betas=(B1, B2))

# move to cuda
encoder.cuda()
decoder.cuda()
discriminator.cuda()
adversarial_loss.cuda()
reconstruction_loss.cuda()

if SEND_WANDB:
    run = wandb.init(
        project="tokai_teio",
        config={
            "learning_rate": LR,
            "epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "latent_dim": LATENT_DIM,
            "img_size": IMG_SIZE,
            "channels": CHANNELS,
            "tags": ["aae", "v0"]
        },
    )

#######################################################
# Training part
#######################################################


def sample_image(n_row, epoch, img_dir):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    z = Variable(Tensor(np.random.normal(0, 1, (n_row**2, LATENT_DIM))))
    generated_imgs = decoder(z)
    save_image(generated_imgs.data, os.path.join(
        img_dir, "%depoch.png" % epoch), nrow=n_row, normalize=True)


for epoch in range(N_EPOCHS):
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
            0, 1, (x.shape[0], LATENT_DIM))))
        real_loss = adversarial_loss(discriminator(real_z), valid)
        fake_loss = adversarial_loss(discriminator(fake_z.detach()), fake)
        D_loss = 0.5*(real_loss + fake_loss)
        D_loss.backward()
        optimizer_D.step()

        if SEND_WANDB:
            wandb.log({"G_loss": G_loss.item(), "D_loss": D_loss.item()})
    print(
        "[Epoch %d/%d] [G loss: %f] [D loss: %f]"
        % (epoch, N_EPOCHS, G_loss.item(), D_loss.item())
    )

    sample_image(n_row=5, epoch=epoch,
                 img_dir=f"outputs/{file_name}")

finished_at = datetime.now().strftime("%Y-%m-%d_%H:%M")
if SEND_WANDB:
    wandb.finish()
torch.save(encoder.state_dict(),
           f'trained_models/encoder/{file_name}_{finished_at}.pth')

torch.save(decoder.state_dict(),
           f'trained_models/decoder/{file_name}_{finished_at}.pth')