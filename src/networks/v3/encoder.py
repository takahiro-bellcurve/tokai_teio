import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_channels, img_size, latent_dim):
        super(Encoder, self).__init__()

        self.input_channels = input_channels
        self.img_size = img_size
        self.latent_dim = latent_dim

        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels, 16, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc1 = nn.Linear(64 * img_size * img_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, self.latent_dim)

    def forward(self, img):
        x = self.features(img)
        x = x.view(x.size(0), -1)  # 特徴をフラット化
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
