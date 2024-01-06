import os
import re
import importlib

import torch

from src.lib.image_preprocessor import ImagePreprocessor


class ModelOperator:
    @staticmethod
    def get_model_name_list(model_type="encoder"):
        model_name_list = []
        for file_name in os.listdir(f"trained_models/{model_type}"):
            model_name = re.sub(r"\.pth", "", file_name)
            model_name_list.append(model_name)
        return model_name_list

    @staticmethod
    def get_model_info_from_model_name(model_name):
        model_info = {}
        model_info["model_name"] = model_name
        model_info["input_channels"] = int(re.search(
            r"ch(\d+)_", model_name).group(1))
        model_info["img_size"] = int(
            re.search(r"_(\d+)_ch", model_name).group(1))
        model_info["latent_dim"] = int(re.search(
            r"ldim_(\d+)_", model_name).group(1))
        model_info["network_version"] = re.search(
            r"(v\d+)_", model_name).group(1)
        print(model_info)
        return model_info

    @staticmethod
    def get_device(force_gpu=False):
        if force_gpu:
            if torch.cuda.is_available() == False:
                raise Exception("No GPU found, please run without --cuda")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    @staticmethod
    def get_encoder_model(network_version: str, input_channels: int, img_size: int, latent_dim: int):
        try:
            network_module = importlib.import_module(
                f"src.networks.{network_version}")
            Encoder = getattr(network_module, 'Encoder')
        except ImportError:
            raise Exception(f"Invalid network version: {network_version}")
        return Encoder(input_channels, img_size, latent_dim)

    @staticmethod
    def get_decoder_model(network_version: str, input_channels: int, img_size: int, latent_dim: int):
        try:
            network_module = importlib.import_module(
                f"src.networks.{network_version}")
            Decoder = getattr(network_module, 'Decoder')
        except ImportError:
            raise Exception(f"Invalid network version: {network_version}")
        return Decoder(input_channels, img_size, latent_dim)

    @staticmethod
    def encode_image(encoder, img_path, img_size, preprocess=False, device=None, to_numpy=True):
        image = ImagePreprocessor.open_image(img_path)
        if preprocess:
            image = ImagePreprocessor.remove_background(image)
            image = ImagePreprocessor.fill_white_background(image)
        image = ImagePreprocessor.resize(
            image, img_size, img_size, to_tensor=True)
        image = image.unsqueeze(0)
        if device is not None:
            image = image.to(device)
        with torch.no_grad():
            vector = encoder(image)

        if to_numpy:
            return vector.cpu().numpy()
        return vector

    @staticmethod
    def save_model(model, file_name, finished_at, model_type="encoder", bucket=None):
        torch.save(model.state_dict(),
                   f'trained_models/{model_type}/{file_name}_{finished_at}.pth')
        if os.getenv("APP_ENV") == "production":
            blob = bucket.blob(
                f'trained_models/{model_type}/{file_name}_{finished_at}.pth')
            blob.upload_from_filename(
                f'trained_models/{model_type}/{file_name}_{finished_at}.pth')
