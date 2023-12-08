import torch
import torch.nn as nn


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
