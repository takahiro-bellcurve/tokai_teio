import os
import re
import logging
from logging import StreamHandler, Formatter

import torch
import pandas as pd
import numpy as np

from src.lib.image_preprocessor import ImagePreprocessor
from src.lib.faiss_operator import FaissOperator

# config
INPUT_CHANNELS = 3
IMG_SIZE = 64
ORIGINAL_IMG_DIR = "./data/preprocessed_images_512/"
TEST_IMG_DIR = "./data/test_images/"

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

# select model
models = os.listdir("./trained_models/encoder/")
for i, model in enumerate(models):
    print(f"{i}: {model}")
model_num = int(input("Select model number: "))
model_name = models[model_num].replace(".pth", "")
input_channels = re.search(r"ch(\d+)_", model_name).group(1)
img_size = re.search(r"_(\d+)_ch", model_name).group(1)
latent_dim = re.search(r"ldim_(\d+)_", model_name).group(1)
network_version = re.search(r"v(\d+)_", model_name).group(1)
if network_version == "v0":
    from src.networks.v0 import Encoder
elif network_version == "v1":
    from src.networks.v1 import Encoder
elif network_version == "v2":
    from src.networks.v2 import Encoder
elif network_version == "v3":
    from src.networks.v3 import Encoder

img_list = os.listdir(ORIGINAL_IMG_DIR)
df = pd.read_csv("./data/zozotown_goods_images_100000.csv")
df["file_name"] = df.apply(create_file_name, axis=1)
faiss_id_mapping_df = delete_unexist_images(df, img_list)
faiss_id_mapping_df.reset_index(drop=True, inplace=True)
faiss_id_mapping_df.to_csv(
    f"./db/faiss/id_mapping/{model_name}.csv", index=False)

model = Encoder(INPUT_CHANNELS, IMG_SIZE, latent_dim)
model.load_state_dict(torch.load(f"trained_models/encoder/{model_name}.pth"))
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
    index, f"db/faiss/{model_name}.index")
logger.info("Finish save faiss index")
