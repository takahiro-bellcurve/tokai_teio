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
from src.lib.model_operator import ModelOperator
from src.lib.create_data_loader import create_data_loader

MODEL_NAME = "aae_v0"
SEND_WANDB = True

#######################################################
# Parameter setting
#######################################################
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20,
                    help="number of epochs of training")
parser.add_argument("--latent_dim", type=int, default=20,
                    help="dimensionality of the latent code")
args = parser.parse_args()

N_EPOCHS = args.n_epochs
BATCH_SIZE = 512
LR = 0.0002
B1 = 0.5
B2 = 0.999
LATENT_DIM = args.latent_dim
IMG_SIZE = 64
CHANNELS = 3
NUM_LABELS = 10

#######################################################
# Preparing part
#######################################################

# config cuda
if not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

if os.getenv("APP_ENV") == "production":
    from google.cloud import storage
    client = storage.Client()
    bucket = client.get_bucket("tokaiteio")
else:
    bucket = None

# define file name
file_name = f"{MODEL_NAME}_{IMG_SIZE}_ch{CHANNELS}_ldim_{LATENT_DIM}_bs_{BATCH_SIZE}_lr_{LR}_b1_{B1}_b2_{B2}_with_gaussian_mixture"

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
    if os.getenv("APP_ENV") == "production":
        blob = bucket.blob(
            f"generated_images/{file_name}/%depoch.png" % epoch)
        blob.upload_from_filename(
            f"generated_images/{file_name}/%depoch.png" % epoch)


def gaussian_mixture(batchsize, ndim, num_labels, device='cpu'):
    if ndim % 2 != 0:
        raise Exception("ndim must be a multiple of 2.")

    def sample(x, y, label, num_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(num_labels) * float(label)
        r_tensor = torch.tensor(r, device=device)
        new_x = x * torch.cos(r_tensor) - y * torch.sin(r_tensor)
        new_y = x * torch.sin(r_tensor) + y * torch.cos(r_tensor)
        new_x += shift * torch.cos(r_tensor)
        new_y += shift * torch.sin(r_tensor)
        return torch.stack([new_x, new_y])

    x_var = 0.5
    y_var = 0.05
    x = torch.normal(0, x_var, (batchsize, ndim // 2)).to(device)
    y = torch.normal(0, y_var, (batchsize, ndim // 2)).to(device)
    z = torch.empty((batchsize, ndim), device=device)
    for batch in range(batchsize):
        for zi in range(ndim // 2):
            label = torch.randint(0, num_labels, (1,)).item()
            z[batch, zi*2:zi*2 +
                2] = sample(x[batch, zi], y[batch, zi], label, num_labels)
    return z


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        a = 0.001
        ad_loss = 0.001 * adversarial_loss(validity_fake_z, valid)
        re_loss = 0.999 * reconstruction_loss(decoded_x, x)
        G_loss = ad_loss + re_loss
        G_loss.backward()
        optimizer_G.step()

        # 2) discriminator loss
        optimizer_D.zero_grad()
        real_z = Variable(Tensor(gaussian_mixture(
            x.shape[0], LATENT_DIM, NUM_LABELS, device=device)).to(x.device))
        real_loss = adversarial_loss(discriminator(real_z), valid)
        fake_loss = adversarial_loss(discriminator(fake_z.detach()), fake)
        D_loss = 0.5*(real_loss + fake_loss)
        D_loss.backward()
        optimizer_D.step()

        if SEND_WANDB:
            wandb.log({"G_loss": G_loss.item(), "D_loss": D_loss.item(),
                       "adversarial_loss": ad_loss.item(), "reconstruction_loss": re_loss.item()})
    print(
        "[Epoch %d/%d] [G loss: %f] [D loss: %f]"
        % (epoch, N_EPOCHS, G_loss.item(), D_loss.item())
    )

    sample_image(n_row=5, epoch=epoch,
                 img_dir=f"generated_images/{file_name}")

finished_at = datetime.now().strftime("%Y-%m-%d_%H:%M")
if SEND_WANDB:
    wandb.finish()

# save model
ModelOperator.save_model(encoder, file_name, finished_at,
                         model_type="encoder", bucket=bucket)
ModelOperator.save_model(decoder, file_name, finished_at,
                         model_type="decoder", bucket=bucket)
ModelOperator.save_model(discriminator, file_name, finished_at,
                         model_type="discriminator", bucket=bucket)
