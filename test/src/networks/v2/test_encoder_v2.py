import torch
from src.networks.v2 import Encoder


def test_encoder_v2_output_shape():
    input_channels = 3
    img_size = 64
    latent_dim = 100

    encoder = Encoder(input_channels, img_size, latent_dim)

    test_input = torch.randn(1, input_channels, img_size,
                             img_size)

    with torch.no_grad():
        output = encoder(test_input)

    assert output.shape == (
        1, latent_dim), f"Output shape is incorrect: expected {(1, latent_dim)}, got {output.shape}"
