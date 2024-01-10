from urllib.parse import urlparse
from pathlib import Path
from time import sleep
import os
import random

import requests
import pandas as pd
from rembg import remove
from PIL import Image

from src.lib.image_preprocessor import ImagePreprocessor


files = os.listdir('data/original_images')
df = pd.read_csv('data/zozotown_goods_images_500000.csv')

df["image_name"] = df["image_url"].apply(lambda x: x.split("/")[-1])


download_df = df[~df['image_name'].isin(files)]
download_df = download_df.reset_index(drop=True)

for i, row in download_df.iterrows():
    if i % 100 == 0:
        print(f"{i} images downloaded")
    image_url = row['image_url']
    image_name = row['image_name']
    try:
        image = ImagePreprocessor.download_image(
            download_df["image_url"][i])
        if image is None:
            continue
        ImagePreprocessor.save_image(
            image, f"./data/original_images/{image_name}")
    except Exception as e:
        print(e)
    sleep(0.1)
