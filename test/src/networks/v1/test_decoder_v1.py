import torch
from src.networks.v1.decoder import Decoder


def test_decoder_v1_output_shape():
    output_channels = 3
    img_size = 64
    latent_dim = 100

    decoder = Decoder(output_channels, img_size, latent_dim)

    test_input = torch.randn(1, latent_dim)

    with torch.no_grad():
        output = decoder(test_input)

    assert output.shape == (1, output_channels, img_size,
                            img_size), "Output shape is incorrect"
