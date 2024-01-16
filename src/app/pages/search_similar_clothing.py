import os
import re
import logging
import subprocess

import streamlit as st
import torch
import pinecone

from src.lib.image_preprocessor import ImagePreprocessor
from src.lib.model_operator import ModelOperator
from src.lib.setup_logging import setup_logging
from src.lib.system_operator import get_gpu_memory_usage

setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Search Similar Image")

st.markdown("# Search Similar Image")

status_text = st.sidebar.empty()

model_file_names = []
for file_name in os.listdir("trained_models/encoder"):
    model_name = re.sub(r"\.pth", "", file_name)
    model_file_names.append(model_name)
model_name = st.selectbox("Select Model", model_file_names)
model_info = ModelOperator.get_model_info_from_model_name(model_name)

st.text(f"input_channels: {model_info['input_channels']}")
st.text(f"img_size: {model_info['img_size']}")
st.text(f"latent_dim: {model_info['latent_dim']}")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_info = ModelOperator.get_model_info_from_model_name(model_name)
    model = ModelOperator.get_encoder_model(
        model_info["network_version"], model_info["input_channels"], model_info["img_size"], model_info["latent_dim"])

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
    vector = encode_image(model, upload_image, model_info["img_size"])
    indexes = index.query(
        vector=vector[0].tolist(), top_k=3, include_metadata=True)

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

    # with col2:
    #     for img in display_images[3:]:
    #         st.image(img)
