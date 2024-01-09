import os
import re
import io
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

    reducer = umap.UMAP(
        random_state=42, n_neighbors=n_neighbors, min_dist=min_dist, spread=spread)
    embedding = reducer.fit_transform(latent_vectors[:data_points])

    fig = plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1], cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the Latent Space')
    buf = io.BytesIO()
    fig.savefig(
        f"./generated_images/umap/{model_name}_nn{n_neighbors}_md{min_dist}_s{spread}.png")
    fig.savefig(buf, format='png')
    buf.seek(0)
    st.session_state["fig"] = buf
    return fig


st.text("data_points: \n データの数")
data_points = st.slider("data_points", 1000, 100000, 10000)
st.text("n_neighbors: \n 値が小さいほど局所的なデータが保持されます")
n_neighbors = st.slider("n_neighbors", 2, 100, 15)
st.text("min_dist: \n 埋め込み空間における点間の最小距離を定義")
min_dist = st.slider("min_dist", 0.0, 1.0, 0.1)
st.text("spread: \n 埋め込み点の効果的なスケール、つまり埋め込まれた点がどの程度広がるかを決定するパラメータ")
spread = st.slider("spread", 0.0, 1.0, 1.0)

if st.button("Show Graph"):
    fig = umap_visualization()
    st.pyplot(fig)


# download graph
if st.session_state.get("fig"):
    if st.download_button(label="Download Graph", data=st.session_state["fig"], file_name=f"{model_name}_nn{n_neighbors}_md{min_dist}_s{spread}.png", mime="image/png"):
        st.success("Saved Graph")
