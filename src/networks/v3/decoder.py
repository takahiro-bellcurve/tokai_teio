from torch import nn


class Decoder(nn.Module):
    def __init__(self, output_channels, img_size, latent_dim):
        super(Decoder, self).__init__()

        self.output_channels = output_channels
        self.img_size = img_size
        self.latent_dim = latent_dim

        # Fully connected layers
        self.fc1 = nn.Linear(self.latent_dim, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 64 * img_size * img_size)

        # Convolutional layers to reshape to original image size
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # Upsample to increase spatial dimensions
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, self.output_channels, 3, stride=1, padding=1),
            nn.Tanh()  # Output layer to match the original image format
        )

    def forward(self, z):
        out = self.fc1(z)
        out = self.fc2(out)
        out = self.fc3(out)
        # Reshape to match the input shape of the first conv layer
        out = out.view(out.size(0), 64, self.img_size, self.img_size)
        img = self.conv_blocks(out)
        return img
