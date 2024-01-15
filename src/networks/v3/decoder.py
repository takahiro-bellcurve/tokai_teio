from torch import nn


class Decoder(nn.Module):
    def __init__(self, output_channels, img_size, latent_dim):
        super(Decoder, self).__init__()

        self.exec_count = False
        self.output_channels = output_channels
        self.img_size = img_size
        self.latent_dim = latent_dim

        self.linear1 = nn.Linear(self.latent_dim, 400)
        self.linear2 = nn.Linear(400, 4000)
        self.linear3 = nn.Linear(4000, 128 * 20 * 20)

        self.deconv1 = nn.ConvTranspose2d(
            128, 64, 5, stride=3, padding=2, output_padding=2)
        self.deconv2 = nn.ConvTranspose2d(
            64, self.output_channels, 5, stride=1, padding=0)

        self.batchnorm = nn.BatchNorm2d(64)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, z):
        print("decoder")
        x = self.linear1(z)
        print(f"linear1: {x.shape}") if self.exec_count == False else None
        x = self.leakyrelu(x)
        x = self.linear2(x)
        print(f"linear2: {x.shape}") if self.exec_count == False else None
        x = self.leakyrelu(x)
        x = self.linear3(x)
        print(f"linear3: {x.shape}") if self.exec_count == False else None
        x = self.leakyrelu(x)

        x = x.view(x.size(0), 128, 20, 20)
        print(
            f"flattened: {x.shape}") if self.exec_count == False else None
        x = self.deconv1(x)
        print(f"deconv1: {x.shape}") if self.exec_count == False else None
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        x = self.deconv2(x)
        print(f"deconv2: {x.shape}") if self.exec_count == False else None
        x = self.tanh(x)  # Often used for normalizing output to [-1, 1]

        self.exec_count = True
        return x
