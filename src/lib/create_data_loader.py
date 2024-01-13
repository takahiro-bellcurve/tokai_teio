import torch
import torchvision.transforms as transforms

from src.lib.image_dataset.image_dataset import ImageDataset


def create_data_loader(img_size, batch_size, channels, dataset_dir='data/train_data'):
    if channels == 3:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
    dataset = ImageDataset(
        directory=dataset_dir, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    return data_loader
