import os
import re
import logging
import subprocess

import torch
import streamlit as st

from src.lib.image_preprocessor import ImagePreprocessor
from src.lib.model_operator import ModelOperator
from src.lib.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Encode Decode Image")

st.markdown("# Encode Decode Image")


# ------------ functions ------------
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


def get_encoder_decoder(image, model_info):
    device = ModelOperator.get_device()
    encoder = ModelOperator.get_encoder_model(
        model_info["network_version"], model_info["input_channels"], model_info["img_size"], model_info["latent_dim"])
    encoder.load_state_dict(torch.load(
        f"trained_models/encoder/{model_info['model_name']}.pth"))
    encoder.to(device)

    decoder = ModelOperator.get_decoder_model(
        model_info["network_version"], model_info["input_channels"], model_info["img_size"], model_info["latent_dim"])
    decoder.load_state_dict(torch.load(
        f"trained_models/decoder/{model_info['model_name']}.pth"))
    decoder.to(device)

    return encoder, decoder


def regenerate_image(model_info, image):
    encoder, decoder = get_encoder_decoder(image, model_info)
    vector = ModelOperator.encode_image(
        encoder, image, model_info["img_size"], to_numpy=False, device=ModelOperator.get_device(), preprocess=True)
    vector = torch.tensor(vector, dtype=torch.float32).to(
        ModelOperator.get_device())
    regenerated_image_vector = decoder(vector)
    regenerated_image = ImagePreprocessor.vector_to_image(
        regenerated_image_vector[0])
    return regenerated_image


# ------------ page ------------
model_file_names = ModelOperator.get_model_name_list()
model_name = st.selectbox("Select Model", model_file_names)

model_info = ModelOperator.get_model_info_from_model_name(model_name)

st.text(f"input_channels: {model_info['input_channels']}")
st.text(f"img_size: {model_info['img_size']}")
st.text(f"latent_dim: {model_info['latent_dim']}")
st.markdown(f"GPUメモリ使用率 {get_gpu_memory_usage()}%")

upload_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
if upload_image is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("## Original Image")
        st.image(upload_image, width=200)
    with col2:
        st.markdown("## preprocessed Image")
        image = ImagePreprocessor.open_image(upload_image)
        image = ImagePreprocessor.remove_background(image)
        image = ImagePreprocessor.fill_white_background(image)
        image = ImagePreprocessor.resize(image, 200, 200)
        st.image(image, width=200)

if upload_image:
    st.divider()
    st.markdown("## Regenerated Image")
    if st.button("Regenerate"):
        regenerated_image = regenerate_image(model_info, upload_image)
        st.image(regenerated_image, width=200)
