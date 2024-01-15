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

# Logger
setup_logging()
logger = logging.getLogger(__name__)


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

    df = pd.read_csv("./data/train_data.csv")
    train_images = os.listdir("./data/train_data")
    df = df[df["file_name"].isin(train_images)]
    image_data = df.to_dict(orient='records')

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
        image_path = "data/train_data/" + row["file_name"]
        vector = ModelOperator.encode_image(model, image_path,
                                            img_size, preprocess=False, device=device)
        record = {
            "image_id": row["image_id"],
            "image_url": row["image_url"],
            "file_name": row["file_name"],
            "category_name": row["category_name"],
            "child_category_name": row["child_category_name"],
            "vector": vector
        }
        vectors.append(record)
    vectors = np.array(vectors).reshape(len(vectors), -1)
    with open(f'./trained_models/vectors/{model_name}.pkl', 'wb') as f:
        pickle.dump(vectors, f)
    logger.info("Finish encoding images")


if __name__ == "__main__":
    main()
