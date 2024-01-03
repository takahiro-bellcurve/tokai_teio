import os
import re
import logging
from logging import StreamHandler, Formatter

import torch
import pandas as pd
import numpy as np
import pinecone

from src.lib.image_preprocessor import ImagePreprocessor


# config
ORIGINAL_IMG_DIR = "./data/preprocessed_images_512/"
PINECONE_INDEX_NAME = "tokai-teio"
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


def encode_image(model, img_path, img_size, preprocess=False, device=None):
    image = ImagePreprocessor.open_image(img_path)
    if preprocess:
        image = ImagePreprocessor.remove_background(image)
        image = ImagePreprocessor.fill_white_background(image)
    image = ImagePreprocessor.resize(image, img_size, img_size, to_tensor=True)
    image = image.unsqueeze(0)
    if device is not None:
        image = image.to(device)
    with torch.no_grad():
        vector = model(image)
    return vector.cpu().numpy()


def main():
    logger.info("Start refresh_pinecone_index.py")
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment='gcp-starter'
    )
    logger.info("delete index")
    try:
        index = pinecone.Index(PINECONE_INDEX_NAME)
        pinecone.delete_index(PINECONE_INDEX_NAME)
        logger.info("index deleted")
    except:
        pass

    # select model
    models = os.listdir("./trained_models/encoder/")
    for i, model in enumerate(models):
        print(f"{i}: {model}")
    model_num = int(input("Select model number: "))
    model_name = models[model_num].replace(".pth", "")
    input_channels = re.search(r"ch(\d+)_", model_name).group(1)
    img_size = re.search(r"_(\d+)_ch", model_name).group(1)
    latent_dim = re.search(r"ldim_(\d+)_", model_name).group(1)
    network_version = re.search(r"(v\d+)_", model_name).group(1)
    print(f"selected model version: {network_version}")
    print(f"selected model name: {model_name}")
    print(f"selected input_channels: {input_channels}")
    print(f"selected img_size: {img_size}")
    print(f"selected latent_dim: {latent_dim}")
    if network_version == "v0":
        from src.networks.v0 import Encoder
    elif network_version == "v1":
        from src.networks.v1 import Encoder
    elif network_version == "v2":
        from src.networks.v2 import Encoder
    elif network_version == "v3":
        from src.networks.v3 import Encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_list = os.listdir(ORIGINAL_IMG_DIR)
    df = pd.read_csv("./data/zozotown_goods_images_100000.csv")
    df["file_name"] = df.apply(create_file_name, axis=1)
    upsert_df = delete_unexist_images(df, img_list)
    upsert_df.reset_index(drop=True, inplace=True)
    upsert_data = upsert_df[['id', 'file_name',
                             'image_url']].to_dict(orient='records')

    model = Encoder(int(input_channels), int(img_size), int(latent_dim))
    model.load_state_dict(torch.load(
        f"trained_models/encoder/{model_name}.pth"))
    model.to(device)
    model.eval()

    logger.info("Start encoding images")
    vectors = []
    for i, row in enumerate(upsert_data):
        if i % 1000 == 0:
            logger.info(f"{i} images encoded")
        image_path = ORIGINAL_IMG_DIR + row["file_name"]
        vector = encode_image(model, image_path, int(
            img_size), preprocess=False, device=device)
        vectors.append(vector)
    vectors = np.array(vectors).reshape(len(vectors), -1)
    logger.info("Finish encoding images")

    upsert_vectors = []
    for i, row in enumerate(upsert_data):
        upsert_vectors.append(
            {
                "id": f"t{str(row['id'])}",
                "values": vectors[i],
                "metadata": {
                    "file_name": row["file_name"],
                    "image_url": row["image_url"]
                }
            }
        )

    logger.info("Start creating faiss index")
    pinecone.create_index(PINECONE_INDEX_NAME, dimension=int(
        latent_dim), metric="euclidean")
    pinecone.describe_index(PINECONE_INDEX_NAME)
    index = pinecone.Index(PINECONE_INDEX_NAME)

    for i in range(0, len(upsert_vectors), 100):
        index.upsert(upsert_vectors[i:i+100])
        logger.info(f"{i} images inserted")
    logger.info("Finish creating faiss index")


if __name__ == "__main__":
    main()
