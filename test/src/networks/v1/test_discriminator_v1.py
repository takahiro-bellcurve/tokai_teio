import torch
from src.networks.v1 import Discriminator


def test_discriminator_v1_output():
    latent_dim = 100
    discriminator = Discriminator(latent_dim)

    test_input = torch.randn(1, latent_dim)

    with torch.no_grad():
        output = discriminator(test_input)

    assert output.size() == (
        1, 1), f"Output size is incorrect: expected {(1, 1)}, got {output.size()}"

    assert 0 <= output.item() <= 1, "Output is out of range (0, 1)"
