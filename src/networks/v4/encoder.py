import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_channels, img_size, latent_dim):
        super(Encoder, self).__init__()
        self.is_fist_execute = False

        self.input_channels = input_channels
        self.img_size = img_size
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(self.input_channels, 64, 5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 128, 5, stride=3, padding=2)
        self.batchnorm = nn.BatchNorm2d(128)

        self.flattened_size = 128 * 20 * 20

        self.linear1 = nn.Linear(self.flattened_size, 8000)
        self.linear2 = nn.Linear(8000, 4000)
        self.linear3 = nn.Linear(4000, 400)
        self.linear4 = nn.Linear(400, self.latent_dim)

        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, img):
        x = self.conv1(img)
        print(f"conv1: {x.shape}") if self.is_fist_execute == False else None
        x = self.leakyrelu(x)
        x = self.conv2(x)
        print(f"conv2: {x.shape}") if self.is_fist_execute == False else None
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        x = x.view(x.size(0), -1)
        print(
            f"flattened: {x.shape}") if self.is_fist_execute == False else None

        x = self.linear1(x)
        print(f"linear1: {x.shape}") if self.is_fist_execute == False else None
        x = self.leakyrelu(x)
        x = self.linear2(x)
        print(f"linear2: {x.shape}") if self.is_fist_execute == False else None
        x = self.leakyrelu(x)
        x = self.linear3(x)
        print(f"linear3: {x.shape}") if self.is_fist_execute == False else None
        x = self.leakyrelu(x)
        x = self.linear4(x)
        print(f"linear4: {x.shape}") if self.is_fist_execute == False else None
        self.is_fist_execute = True
        return x
