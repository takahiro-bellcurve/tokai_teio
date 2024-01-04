import os
import re
import subprocess
import pickle
import umap
import logging
from logging import StreamHandler, Formatter
import matplotlib.pyplot as plt

import streamlit as st

stream_handler = StreamHandler()
stream_handler.setFormatter(Formatter(
    '%(asctime)s [%(name)s] %(levelname)s: %(message)s', datefmt='%Y/%d/%m %I:%M:%S'))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(stream_handler)

st.set_page_config(page_title="Umap Visualization")

st.markdown("# Umap Visualization")

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
