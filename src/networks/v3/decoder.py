from torch import nn


class Decoder(nn.Module):
    def __init__(self, output_channels, img_size, latent_dim):
        super(Decoder, self).__init__()

        self.output_channels = output_channels
        self.img_size = img_size
        self.latent_dim = latent_dim

        # 潜在ベクトルから初期特徴マップサイズへの変換
        self.init_size = img_size // 4  # アップサンプリングのための初期サイズ
        self.l1 = nn.Sequential(
            nn.Linear(self.latent_dim, 1024)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(1024, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.output_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 1024, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
