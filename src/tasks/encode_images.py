import os
import re
import pickle
import argparse
import logging
from time import sleep

import torch
import pandas as pd
import numpy as np

from src.lib.model_operator import ModelOperator
from src.lib.setup_logging import setup_logging


# config
ORIGINAL_IMG_DIR = "./data/preprocessed_images_512/"

# Logger
setup_logging()
logger = logging.getLogger(__name__)


def create_file_name(row):
    return row["image_url"].split("/")[-1]


def delete_unexist_images(df, img_list):
    df = df[df["file_name"].isin(img_list)]
    return df


def main():
    logger.info("start encode_images.py")
    # select model
    models = os.listdir("./trained_models/encoder/")
    for i, model in enumerate(models):
        print(f"{i}: {model}")
    model_num = int(input("Select model number: "))

    model_info = ModelOperator.get_model_info_from_model_name(
        models[model_num])
    network_version = model_info["network_version"]
    model_name = re.sub(r"\.pth", "", models[model_num])
    input_channels = model_info["input_channels"]
    img_size = model_info["img_size"]
    latent_dim = model_info["latent_dim"]

    img_list = os.listdir(ORIGINAL_IMG_DIR)
    df = pd.read_csv("./data/zozotown_goods_images_100000.csv")
    df["file_name"] = df.apply(create_file_name, axis=1)
    df = delete_unexist_images(df, img_list)
    df.reset_index(drop=True, inplace=True)
    image_data = df[['id', 'file_name',
                     'image_url']].to_dict(orient='records')

    device = ModelOperator.get_device()
    model = ModelOperator.get_encoder_model(
        network_version, input_channels, img_size, latent_dim)
    model.load_state_dict(torch.load(
        f"trained_models/encoder/{model_name}.pth"))
    model.to(device)
    model.eval()

    logger.info("Start encoding images")
    vectors = []
    for i, row in enumerate(image_data):
        if i % 1000 == 0:
            logger.info(f"{i} images encoded")
        image_path = ORIGINAL_IMG_DIR + row["file_name"]
        vector = ModelOperator.encode_image(model, image_path,
                                            img_size, preprocess=False, device=device)
        vectors.append(vector)
    vectors = np.array(vectors).reshape(len(vectors), -1)
    with open(f'./trained_models/vectors/{model_name}.pkl', 'wb') as f:
        pickle.dump(vectors, f)
    logger.info("Finish encoding images")


if __name__ == "__main__":
    main()
