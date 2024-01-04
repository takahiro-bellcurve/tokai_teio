import os
import re
import logging
from logging import StreamHandler, Formatter
import subprocess

import streamlit as st
import torch
import pinecone

from src.lib.image_preprocessor import ImagePreprocessor

stream_handler = StreamHandler()
stream_handler.setFormatter(Formatter(
    '%(asctime)s [%(name)s] %(levelname)s: %(message)s', datefmt='%Y/%d/%m %I:%M:%S'))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(stream_handler)

st.set_page_config(page_title="Search Similar Image")

st.markdown("# Search Similar Image")

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


model_file_names = []
for file_name in os.listdir("trained_models/encoder"):
    model_name = re.sub(r"\.pth", "", file_name)
    model_file_names.append(model_name)
model_name = st.selectbox("Select Model", model_file_names)
input_channels = re.search(r"ch(\d+)_", model_name).group(1)
img_size = re.search(r"_(\d+)_ch", model_name).group(1)
latent_dim = re.search(r"ldim_(\d+)_", model_name).group(1)

st.text(f"input_channels: {input_channels}")
st.text(f"img_size: {img_size}")
st.text(f"latent_dim: {latent_dim}")
st.markdown(f"GPUメモリ使用率 {get_gpu_memory_usage()}%")

upload_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
if upload_image is not None:
    st.image(upload_image, width=200)


def encode_image(model, image, img_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = ImagePreprocessor.open_image(upload_image)
    image = ImagePreprocessor.remove_background(image)
    image = ImagePreprocessor.fill_white_background(image)
    image = ImagePreprocessor.resize(image, img_size, img_size, to_tensor=True)
    image = image.unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        vector = model(image)
    return vector.cpu().numpy()


def search_similar_image(upload_image, model_name):
    logger.info("Start searching similar image")
    print("Start searching similar image")
    st.spinner("Searching Similar Image...")

    input_channels = re.search(r"ch(\d+)_", model_name).group(1)
    img_size = re.search(r"_(\d+)_ch", model_name).group(1)
    latent_dim = re.search(r"ldim_(\d+)_", model_name).group(1)
    network_version = re.search(r"v(\d+)_", model_name).group(1)
    if network_version == "0":
        from src.networks.v0.encoder import Encoder
    elif network_version == "1":
        from src.networks.v1.encoder import Encoder
    elif network_version == "2":
        from src.networks.v2.encoder import Encoder
    elif network_version == "3":
        from src.networks.v3.encoder import Encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Encoder(int(input_channels), int(img_size), int(latent_dim))
    logger.info(f"Loading Encoder: {model_name}")
    model.load_state_dict(torch.load(
        f"trained_models/encoder/{model_name}.pth"))
    logger.info("Encoder Loaded")
    model.to(device)
    model.eval()

    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"], environment="gcp-starter"
    )
    index = pinecone.Index("tokai-teio")
    vector = encode_image(model, upload_image, int(img_size))
    indexes = index.query(
        vector=vector[0].tolist(), top_k=6, include_metadata=True)

    similar_image_urls = []
    for row in indexes["matches"]:
        similar_image_urls.append(row.metadata["image_url"])
    st.session_state["similar_images"] = similar_image_urls


if upload_image:
    st.divider()
    st.markdown("## Search Similar Images")
    if st.button("Search Similar Image"):
        search_similar_image(upload_image, model_name)

if "similar_images" in st.session_state:
    col1, col2 = st.columns(2)
    display_images = st.session_state["similar_images"]
    with col1:
        for img in display_images[:3]:
            st.image(img)

    with col2:
        for img in display_images[3:]:
            st.image(img)
