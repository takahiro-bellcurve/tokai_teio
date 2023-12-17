import os
import logging
from logging import StreamHandler, Formatter

import torch
import pandas as pd
import numpy as np

from src.networks.v2.encoder import Encoder
from src.lib.image_preprocessor import ImagePreprocessor
from src.lib.faiss_operator import FaissOperator

# config
INPUT_CHANNELS = 3
IMG_SIZE = 64
LATENT_DIM = 30
ORIGINAL_IMG_DIR = "./data/preprocessed_images_512/"
TEST_IMG_DIR = "./data/test_images/"
ENCODER_MODEL_NAME = "aae_v0_64_ch3_ldim_30_bs_512_lr_0.0002_b1_0.5_b2_0.999_2023-12-16_23:37"

# Logger
stream_handler = StreamHandler()
stream_handler.setFormatter(Formatter(
    '%(asctime)s [%(name)s] %(levelname)s: %(message)s', datefmt='%Y/%d/%m %I:%M:%S'))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(stream_handler)


def create_file_name(row):
    return row["image_url"].split("/")[-1]


def delete_unexist_images(df, img_list):
    df = df[df["file_name"].isin(img_list)]
    return df


def encode_image(model, image_path, preprocess=False, device=None):
    image = ImagePreprocessor.open_image(image_path)
    if preprocess:
        image = ImagePreprocessor.remove_background(image)
        image = ImagePreprocessor.fill_white_background(image)
    image = ImagePreprocessor.resize(image, IMG_SIZE, IMG_SIZE, to_tensor=True)
    image = image.unsqueeze(0)
    if device is not None:
        image = image.to(device)
    with torch.no_grad():
        vector = model(image)
    return vector.cpu().numpy()


logger.info("Start faiss_create_index.py")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_list = os.listdir(ORIGINAL_IMG_DIR)
df = pd.read_csv("./data/zozotown_goods_images_100000.csv")
df["file_name"] = df.apply(create_file_name, axis=1)
faiss_id_mapping_df = delete_unexist_images(df, img_list)
faiss_id_mapping_df.reset_index(drop=True, inplace=True)
faiss_id_mapping_df.to_csv(
    f"./db/faiss/id_mapping/{ENCODER_MODEL_NAME}.csv", index=False)

model = Encoder(INPUT_CHANNELS, IMG_SIZE, LATENT_DIM)
model = torch.load(f"trained_models/encoder/{ENCODER_MODEL_NAME}.pth")
model.to(device)
model.eval()

logger.info("Start encoding images")
vectors = []
for file_name in list(faiss_id_mapping_df["file_name"]):
    if len(vectors) % 1000 == 0:
        logger.info(f"{len(vectors)} images encoded")
    image_path = ORIGINAL_IMG_DIR + file_name
    vector = encode_image(model, image_path, preprocess=False, device=device)
    vectors.append(vector)
logger.info("Finish encoding images")

logger.info("Start creating faiss index")
vectors = np.array(vectors).reshape(len(faiss_id_mapping_df), -1)
index = FaissOperator.create_faiss_index(vectors, with_cuda=True)
logger.info("Finish creating faiss index")

FaissOperator.save_faiss_index(
    index, f"db/faiss/{ENCODER_MODEL_NAME}.index")
logger.info("Finish save faiss index")
