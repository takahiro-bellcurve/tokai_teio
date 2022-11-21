FROM nvidia/cuda:9.0-base

# Minicondaをインストールするための最低限のパッケージをインストール
RUN set -ex \
    && deps=' \
        bzip2 \
        ca-certificates \
        curl \
        libgomp1 \
        libgfortran3 \
    ' \
    && apt-get update \
    && apt-get install -y --no-install-recommends $deps \
    && rm -rf /var/lib/apt/lists/*

ENV PKG_URL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
ENV INSTALLER miniconda.sh

# minicondaをインストール
RUN set -ex \
    && curl -kfSL $PKG_URL -o $INSTALLER \
    && chmod 755 $INSTALLER \
    && ./$INSTALLER -b -p /opt/conda3 \
    && rm $INSTALLER

# minicondaをPATHに追加
ENV PATH /opt/conda3/bin:$PATH

# PyTorch v0.4をインストール
ENV PYTORCH_VERSION 0.4

RUN set -ex \
    && pkgs=" \
        pytorch=${PYTORCH_VERSION} \
        torchvision \
    " \
    && conda install -y ${pkgs} -c pytorch \
    && conda clean -i -l -t -y
