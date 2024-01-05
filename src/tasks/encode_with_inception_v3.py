import os
import pickle
import re
import argparse
import logging
from logging import StreamHandler, Formatter
from time import sleep

import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from PIL import Image
import pandas as pd
import numpy as np
import pinecone


from src.lib.model_operator import ModelOperator

# config
ORIGINAL_IMG_DIR = "./data/preprocessed_images_512/"
PINECONE_INDEX_NAME = "tokai-teio"

# Logger
stream_handler = StreamHandler()
stream_handler.setFormatter(Formatter(
    '%(asctime)s [%(name)s] %(levelname)s: %(message)s', datefmt='%Y/%m/%d %I:%M:%S'))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(stream_handler)

parser = argparse.ArgumentParser()
parser.add_argument("--encode", type=int, default=1,
                    help="encode images")
args = parser.parse_args()


def create_file_name(row):
    return row["image_url"].split("/")[-1]


def delete_unexist_images(df, img_list):
    df = df[df["file_name"].isin(img_list)]
    return df


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

# 画像の前処理関数


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(299),  # Inception v3の入力サイズに合わせる
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # バッチ次元の追加
    return image


def main():
    logger.info("Start refresh_pinecone_index.py")
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY_2"),
        environment='gcp-starter'
    )
    logger.info("delete index")

    try:
        index = pinecone.Index(PINECONE_INDEX_NAME)
        pinecone.delete_index(PINECONE_INDEX_NAME)
        logger.info("index deleted")
        sleep(2)
    except:
        pass
    logger.info("Start creating pinecone index")
    pinecone.create_index(PINECONE_INDEX_NAME,
                          dimension=1000, metric="euclidean")
    pinecone.describe_index(PINECONE_INDEX_NAME)

    img_list = os.listdir(ORIGINAL_IMG_DIR)
    df = pd.read_csv("./data/zozotown_goods_images_100000.csv")
    df["file_name"] = df.apply(create_file_name, axis=1)
    upsert_df = delete_unexist_images(df, img_list)
    upsert_df.reset_index(drop=True, inplace=True)
    upsert_data = upsert_df[['id', 'file_name',
                            'image_url']].to_dict(orient='records')
    upsert_data = upsert_data[:10000]

    logger.info(f"encode process is {args.encode}")
    if args.encode == 1:
        device = ModelOperator.get_device()
        # Inception v3モデルのロード
        model = inception_v3(pretrained=True)
        model.to(device)
        model.eval()

        logger.info("Start encoding images")
        vectors = []
        for i, row in enumerate(upsert_data):
            if i % 1000 == 0:
                logger.info(f"{i} images encoded")
            image_path = ORIGINAL_IMG_DIR + row["file_name"]
            image = preprocess_image(image_path)
            image = image.to(device)
            # 画像をモデルに通す
            with torch.no_grad():
                outputs = model(image)
            # 特徴ベクトルの取得
            vector = outputs.detach()[0]
            vectors.append(vector)
        # cpuに移動してnumpyに変換
        vectors = [vector.cpu().numpy() for vector in vectors]
        with open(f'./trained_models/vectors/inception_v3.pkl', 'wb') as f:
            pickle.dump(vectors, f)
        logger.info("Finish encoding images")

    else:
        with open(f'./trained_models/vectors/inception_v3.pkl', 'rb') as f:
            vectors = pickle.load(f)

    upsert_vectors = []
    for i, row in enumerate(upsert_data):
        upsert_vectors.append(
            {
                "id": f"t{str(row['id'])}",
                "values": vectors[i][0],
                "metadata": {
                    "file_name": row["file_name"],
                    "image_url": row["image_url"]
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
