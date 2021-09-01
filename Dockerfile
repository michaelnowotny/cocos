# Based on the official ArrayFire Dockerfile at `https://github.com/arrayfire/arrayfire-docker`
FROM nvidia/cuda:11.2.0-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

#        libopenblas-dev \
#        libatlas-base-dev \

RUN apt-get update && apt-get install -y software-properties-common && \
    apt-get install -y --no-install-recommends \
        build-essential \
        clinfo \
        cmake \
        git \
        intel-mkl-full \
        libboost-all-dev \
        libfftw3-dev \
        libfontconfig1-dev \
        libfreeimage-dev \
        liblapack-dev \
        liblapacke-dev \
        libtbb-dev \
        ocl-icd-opencl-dev \
        opencl-headers \
        python3-pip \
        wget \
        xorg-dev
#    rm -rf /var/lib/apt/lists/*

RUN apt install -y gnupg2 ca-certificates apt-utils

# Setting up symlinks for libcuda and OpenCL ICD
#RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/libcuda.so.1 && \
#    ln -s /usr/lib/libcuda.so.1 /usr/lib/libcuda.so && \
# RUN echo "/usr/local/cuda/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
#RUN mkdir -p /etc/OpenCL/vendors && \
#    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd && \

RUN echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/nvidia.conf
RUN ldconfig
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/nvvm/lib64:$LD_LIBRARY_PATH

# Add ArrayFire Repo and Install
#RUN apt-key adv --fetch-key https://repo.arrayfire.com/GPG-PUB-KEY-ARRAYFIRE-2020.PUB
#RUN echo "deb [arch=amd64] https://repo.arrayfire.com/ubuntu focal main" | tee /etc/apt/sources.list.d/arrayfire.list
#RUN apt-get update && apt-get install -y --no-install-recommends arrayfire arrayfire-dev
#RUN ldconfig

WORKDIR /root

## Add MKL
#RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
#RUN apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
#RUN wget https://apt.repos.intel.com/setup/intelproducts.list -O /etc/apt/sources.list.d/intelproducts.list
#RUN apt-get update
#RUN apt-get install intel-mkl-64bit-2018.2-046
#RUN echo "/opt/intel/lib/intel64"     >  /etc/ld.so.conf.d/mkl.conf && \
#    echo "/opt/intel/mkl/lib/intel64" >> /etc/ld.so.conf.d/mkl.conf && \
#    ldconfig

# Build GLFW from source
RUN git clone https://github.com/glfw/glfw.git && \
    cd glfw && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr .. && \
    make -j4 && \
    make install


# AF_DISABLE_GRAPHICS - Environment variable to disable graphics at
# runtime due to lack of graphics support by docker - visit
# http://arrayfire.org/docs/configuring_environment.htm#af_disable_graphics
# for more information
ENV AF_PATH=/opt/arrayfire AF_DISABLE_GRAPHICS=1
ARG COMPILE_GRAPHICS=OFF

RUN git clone --recursive https://github.com/arrayfire/arrayfire.git -b master && \
    cd arrayfire && mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/opt/arrayfire-3 \
             -DCMAKE_BUILD_TYPE=Release \
             -DUSE_CPU_MKL=OFF  \
             -DUSE_OPENCL_MKL=OFF
             -DBUILD_CPU=OFF  \
             -DBUILD_CUDA=ON  \
             -DBUILD_OPENCL=OFF  \
             -DBUILD_UNIFIED=ON \
             -DBUILD_GRAPHICS=${COMPILE_GRAPHICS} \
             -DBUILD_NONFREE=OFF \
             -DBUILD_EXAMPLES=OFF \
             -DBUILD_TEST=ON \
             -DBUILD_DOCS=OFF \
             -DINSTALL_FORGE_DEV=OFF \
             -DUSE_FREEIMAGE_STATIC=OFF && \
    make -j8 && make install && \
    mkdir -p ${AF_PATH} && ln -s /opt/arrayfire-3/* ${AF_PATH}/ && \
    echo "${AF_PATH}/lib" >> /etc/ld.so.conf.d/arrayfire.conf && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/arrayfire.conf && \
    echo "/usr/local/cuda/nvvm/lib64" >> /etc/ld.so.conf.d/arrayfire.conf && \
    ldconfig

#    cmake .. -DCMAKE_INSTALL_PREFIX=/opt/arrayfire-3 \
#             -DCMAKE_BUILD_TYPE=Release \
#             -DBUILD_CPU=ON \
#             -DBUILD_CUDA=ON \
#             -DBUILD_OPENCL=OFF \
#             -DBUILD_UNIFIED=ON \
#             -DBUILD_GRAPHICS=${COMPILE_GRAPHICS} \
#             -DBUILD_NONFREE=OFF \
#             -DBUILD_EXAMPLES=OFF \
#             -DBUILD_TEST=ON \
#             -DBUILD_DOCS=OFF \
#             -DINSTALL_FORGE_DEV=OFF \
#             -DUSE_FREEIMAGE_STATIC=OFF && \
#             # -DCOMPUTES_DETECTED_LISTinstal="30;35;37;50;52;60" \
#    make -j8 && make install && \
#    mkdir -p ${AF_PATH} && ln -s /opt/arrayfire-3/* ${AF_PATH}/ && \
#    echo "${AF_PATH}/lib" >> /etc/ld.so.conf.d/arrayfire.conf && \
#    echo "/usr/local/cuda/nvvm/lib64" >> /etc/ld.so.conf.d/arrayfire.conf && \
#    ldconfig

WORKDIR /root/arrayfire

#RUN git clone https://github.com/michaelnowotny/cocos.git && \
#    cd /cocos && \
#    pip3 install .

ENTRYPOINT bash
