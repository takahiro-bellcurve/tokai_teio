import os
import re
import logging
from logging import StreamHandler, Formatter
import subprocess
import gc

import streamlit as st
import torch
import pandas as pd

from src.lib.image_preprocessor import ImagePreprocessor
from src.lib.faiss_operator import FaissOperator

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
st.selectbox("Select Model", model_file_names)
input_channels = re.search(r"ch(\d+)_", model_name).group(1)
img_size = re.search(r"_(\d+)_ch", model_name).group(1)
latent_dim = re.search(r"ldim_(\d+)_", model_name).group(1)

st.text(f"input_channels: {input_channels}")
st.text(f"img_size: {img_size}")
st.text(f"latent_dim: {latent_dim}")
st.markdown(f"GPUメモリ使用率 {get_gpu_memory_usage()}%")


def load_faiss_and_model(model_name):
    if "has_setted_up" in st.session_state:
        return False
    st.session_state["has_setted_up"] = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.session_state['device'] = device

    db = FaissOperator.load_faiss_index(
        f"./db/faiss/{model_name}.index", with_cuda=True)
    st.session_state['faiss_index'] = db
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

    model = Encoder(int(input_channels), int(img_size), int(latent_dim))
    logger.info(f"Loading Encoder: {model_name}")
    model.load_state_dict(torch.load(f"trained_models/encoder/{model_name}.pth"))
    logger.info("Encoder Loaded")
    model.to(device)
    model.eval()
    st.session_state['model'] = model
    return True


def check_faiss_and_model():
    return "has_setted_up" in st.session_state


bt1, bt2, bt3 = st.columns(3)

with bt1:
    if st.button("Load Faiss Index and Model", type="primary"):
        status = load_faiss_and_model(model_name)
        if status:
            st.success("Loaded Faiss Index and Model")
        else:
            st.warning("Already loaded Faiss Index and Model")

with bt2:
    if st.button("Check Faiss Index and Model"):
        status = check_faiss_and_model()
        if status:
            st.success("Already loaded Faiss Index and Model")
        else:
            st.warning("Not loaded Faiss Index and Model")


with bt3:
    if st.button("Clear Session State"):
        st.session_state.clear()
        torch.cuda.empty_cache()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Cleared Session State")

upload_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
if upload_image is not None:
    st.image(upload_image, width=200)


def encode_image(model, image, img_size):
    device = st.session_state['device']
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
    if "model" not in st.session_state:
        st.error("Load Faiss Index and Modelを最初に実行してください。")
        return
    logger.info("Start searching similar image")
    print("Start searching similar image")
    st.spinner("Searching Similar Image...")
    model = st.session_state['model']
    db = st.session_state['faiss_index']
    vector = encode_image(model, upload_image, int(img_size))
    indexes = FaissOperator.search_similar_images(db, vector, k=6)

    df = pd.read_csv(f"./db/faiss/id_mapping/{model_name}.csv")
    searched_df = df.iloc[indexes[0]]
    st.session_state["similar_images"] = searched_df["image_url"].values.tolist()


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
