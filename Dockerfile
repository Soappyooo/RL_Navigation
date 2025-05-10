FROM ubuntu:24.04
WORKDIR /app
SHELL ["/bin/bash", "-c", "-i"]

# get basic dependencies
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    curl wget gcc g++ ffmpeg cmake ca-certificates git mesa-utils && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# install conda
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    source /root/miniconda3/bin/activate && \
    conda init bash && \
    conda config --set auto_activate_base false

# set conda env variables
ENV PATH="/root/miniconda3/bin:${PATH}" CONDA_ROOT="/root/miniconda3"

ENV LD_LIBRARY_PATH=/usr/lib/wsl/lib \
    LIBVA_DRIVER_NAME=d3d12 \
    MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA

# create python environment and install packages
RUN conda create -n mine_env python=3.8 -y && \
    conda activate mine_env && \
    conda install -c conda-forge libstdcxx-ng -y && \
    pip install setuptools==65.5.0 pip==21 && \
    pip install wheel==0.38.0 && \
    pip install mlagents-envs opencv-python==4.5.5.64

RUN conda activate mine_env && \
    pip install stable-baselines3==1.5.0 gym numpy torch accelerate && \
    pip install protobuf~=3.20 tensorboard ipykernel pynput

# Add color prompt to bashrc
RUN echo 'PS1="\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]# "' >> /root/.bashrc



ENTRYPOINT ["/bin/bash", "-c", "-i"]

