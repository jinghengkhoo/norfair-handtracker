#
# Based on https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1786#issuecomment-1180161510
# but changing base image.
#
# FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
FROM nvcr.io/nvidia/tensorrt:22.09-py3
# https://forums.developer.nvidia.com/t/issues-building-docker-image-from-ngc-container-nvcr-io-nvidia-pytorch-22-py3/209034/5
ENV PATH="${PATH}:/opt/hpcx/ompi/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib"

# Install dependencies for OpenPose and its installation
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update --yes && apt-get install --yes --auto-remove \
    git \
    ffmpeg libsm6 libxext6 \
    python3-dev \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Pip dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
    numpy \
    opencv-python

RUN pip install git+https://github.com/tryolabs/norfair.git@master

WORKDIR /demo/src

COPY requirements.txt .

RUN python3 -m pip install -r requirements.txt