import os
import re
import pickle
import argparse
import logging
from time import sleep


import torch
import pandas as pd
import numpy as np
import pinecone

from src.lib.model_operator import ModelOperator
from src.lib.setup_logging import setup_logging

# config
PINECONE_INDEX_NAME = "tokai-teio"
# Logger
setup_logging()
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--encode", type=int, default=1,
                    help="encode images")
args = parser.parse_args()


def upsert_vectors_to_pinecone(index, upsert_vectors, retry_count=0):
    try:
        index.upsert(upsert_vectors)
        sleep(0.2)
        return "success"
    except Exception as e:
        logger.warning(e)
        sleep(5)
        retry_count += 1
        logger.info(f"retry {retry_count} times")
        if retry_count > 5:
            return "failed"
        upsert_vectors_to_pinecone(index, upsert_vectors, retry_count)


def main():
    logger.info("Start refresh_pinecone_index.py")
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment='gcp-starter'
    )
    logger.info("delete index")

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

    try:
        index = pinecone.Index(PINECONE_INDEX_NAME)
        pinecone.delete_index(PINECONE_INDEX_NAME)
        logger.info("index deleted")
        sleep(2)
    except:
        pass
    logger.info("Start creating pinecone index")
    pinecone.create_index(PINECONE_INDEX_NAME, dimension=int(
        latent_dim), metric="euclidean")
    pinecone.describe_index(PINECONE_INDEX_NAME)

    df = pd.read_csv("./data/train_data.csv")
    train_images = os.listdir("./data/train_data")
    df = df[df["file_name"].isin(train_images)]
    upsert_data = df.to_dict(orient='records')

    logger.info(f"encode process is {args.encode}")
    if args.encode == 1:
        device = ModelOperator.get_device()
        model = ModelOperator.get_encoder_model(
            network_version, input_channels, img_size, latent_dim)
        model.load_state_dict(torch.load(
            f"trained_models/encoder/{model_name}.pth"))
        model.to(device)
        model.eval()

        logger.info("Start encoding images")
        vectors = []
        for i, row in enumerate(upsert_data):
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

    else:
        with open(f'./trained_models/vectors/{model_name}.pkl', 'rb') as f:
            vectors = pickle.load(f)

    upsert_vectors = []
    for i, row in enumerate(vectors):
        row = row[0]
        upsert_vectors.append(
            {
                "id": row['image_id'],
                "values": row["vector"][0],
                "metadata": {
                    "image_url": row["image_url"],
                    "category_name": row["category_name"],
                    "child_category_name": row["child_category_name"],
                }
            }
        )

    index = pinecone.Index(PINECONE_INDEX_NAME)

    for i in range(0, len(upsert_vectors), 100):
        result = upsert_vectors_to_pinecone(index, upsert_vectors[i:i+100])
        if result == "success":
            logger.info(f"{i} images inserted")
        else:
            logger.warning("upsert failed")
            break
        sleep(1)
    logger.info("Finish refresh_pinecone_index.py")


if __name__ == "__main__":
    main()
