# Based on the official ArrayFire Dockerfile at `https://github.com/arrayfire/arrayfire-docker`
FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04
CMD nvidia-smi

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y software-properties-common && \
    apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        ca-certificates \
        clinfo \
        cmake \
        git \
        gnupg2 \
        libboost-all-dev \
        opencl-headers \
        ocl-icd-opencl-dev \
        python3-pip \
        wget \
        xorg-dev && \
    rm -rf /var/lib/apt/lists/*

# Add Shared CUDA Libraries
RUN echo "/usr/local/cuda/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/nvidia.conf
RUN ldconfig
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/nvvm/lib64:$LD_LIBRARY_PATH

# Setting up symlinks for libcuda and OpenCL ICD
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/libcuda.so.1 && \
    ln -s /usr/lib/libcuda.so.1 /usr/lib/libcuda.so
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Add ArrayFire Repo and Install
RUN apt-key adv --fetch-key https://repo.arrayfire.com/GPG-PUB-KEY-ARRAYFIRE-2020.PUB
RUN echo "deb [arch=amd64] https://repo.arrayfire.com/ubuntu focal main" | tee /etc/apt/sources.list.d/arrayfire.list
RUN apt-get update && apt-get install -y --no-install-recommends arrayfire arrayfire-dev
RUN ldconfig

WORKDIR /root

# Install Cocos
RUN git clone https://github.com/michaelnowotny/cocos.git && \
    cd cocos && \
    pip3 install .

WORKDIR /root/cocos

ENTRYPOINT bash
