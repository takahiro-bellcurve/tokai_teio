import os
import re
import logging
import subprocess

import streamlit as st
import torch
import pinecone
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from PIL import Image

from src.lib.image_preprocessor import ImagePreprocessor
from src.lib.model_operator import ModelOperator
from src.lib.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Search Similar Image with Inception v3")

st.markdown("# Search Similar Image with Inception v3")

status_text = st.sidebar.empty()


def get_gpu_memory_usage():
    # nvidia-smi コマンドを実行
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"])
    # 出力をデコード
    result = result.decode('utf-8')

    # GPUメモリの使用量と総メモリを抽出
    gpu_memory = [list(map(int, re.findall(r'\d+', line)))
                  for line in result.strip().split('\n')]

    # 使用率を計算
    usage = [(used / total) * 100 for used, total in gpu_memory]
    return str(usage[0])[:5]


st.markdown(f"GPUメモリ使用率 {get_gpu_memory_usage()}%")

upload_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
if upload_image is not None:
    st.image(upload_image, width=200)


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(299),  # Inception v3の入力サイズに合わせる
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    image = Image.open(image)
    image = transform(image).unsqueeze(0)  # バッチ次元の追加
    return image


def search_similar_image(upload_image):
    logger.info("Start searching similar image")
    print("Start searching similar image")
    st.spinner("Searching Similar Image...")

    device = ModelOperator.get_device()
    # Inception v3モデルのロード
    model = inception_v3(pretrained=True)
    model.to(device)
    model.eval()

    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY_2"], environment="gcp-starter"
    )
    index = pinecone.Index("tokai-teio")
    # 画像をモデルに通す
    image = preprocess_image(upload_image)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
    # 特徴ベクトルの取得
    vector = outputs.detach()[0]
    vector = vector.cpu().numpy().tolist()
    indexes = index.query(
        vector=vector, top_k=6, include_metadata=True)

    similar_image_urls = []
    for row in indexes["matches"]:
        similar_image_urls.append(row.metadata["image_url"])
    st.session_state["similar_images"] = similar_image_urls


if upload_image:
    st.divider()
    st.markdown("## Search Similar Images")
    if st.button("Search Similar Image"):
        search_similar_image(upload_image)

if "similar_images" in st.session_state:
    col1, col2 = st.columns(2)
    display_images = st.session_state["similar_images"]
    with col1:
        for img in display_images[:3]:
            st.image(img)

    with col2:
        for img in display_images[3:]:
            st.image(img)
