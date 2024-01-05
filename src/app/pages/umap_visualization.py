import os
import re
import subprocess
import pickle
import umap
import logging
import matplotlib.pyplot as plt

import streamlit as st

from src.lib.setup_logging import setup_logging
from src.lib.system_operator import get_gpu_memory_usage

setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Umap Visualization")

st.markdown("# Umap Visualization")

status_text = st.sidebar.empty()

model_file_names = []
for file_name in os.listdir("trained_models/vectors"):
    model_name = re.sub(r"\.pkl", "", file_name)
    model_file_names.append(model_name)
model_name = st.selectbox("Select Model", model_file_names)
input_channels = re.search(r"ch(\d+)_", model_name).group(1)
img_size = re.search(r"_(\d+)_ch", model_name).group(1)
latent_dim = re.search(r"ldim_(\d+)_", model_name).group(1)

st.text(f"input_channels: {input_channels}")
st.text(f"img_size: {img_size}")
st.text(f"latent_dim: {latent_dim}")
st.markdown(f"GPUメモリ使用率 {get_gpu_memory_usage()}%")


def umap_visualization():
    # pickleファイルのパス
    pickle_file = f'trained_models/vectors/{model_name}.pkl'

    # pickleファイルの読み込み
    with open(pickle_file, 'rb') as f:
        latent_vectors = pickle.load(f)

    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(latent_vectors)

    fig = plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1], cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the Latent Space')
    return fig


if st.button("Show Graph"):
    st.pyplot(umap_visualization())
