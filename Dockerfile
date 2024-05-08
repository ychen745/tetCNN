FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV TZ=US
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git python3 python3-distutils wget build-essential vim
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3 get-pip.py

RUN pip install Cython numpy scipy pandas scikit-image scikit-learn
RUN pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

RUN pip install torch_geometric
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

WORKDIR /