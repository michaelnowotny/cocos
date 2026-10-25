# syntax=docker/dockerfile:1
#
# Cocos on Ubuntu 24.04 with ArrayFire 3.10 (CUDA, OpenCL and CPU backends).
#
#   docker build -t cocos .
#   docker run --rm -it --gpus all cocos
#
# GPU access needs an NVIDIA driver >= 525.60 and the NVIDIA Container Toolkit
# on the host. The image contains no CUDA toolkit: ArrayFire bundles the CUDA
# 12.8 runtime libraries it uses, and the driver is provided by the host.
#
# The build downloads the 3.9 GB ArrayFire installer. To reuse a local copy,
# put it in a directory of its own and pass that directory as a build context:
#
#   docker build --build-context arrayfire-installer=/path/to/dir -t cocos .

ARG UBUNTU_VERSION=24.04
ARG AF_VERSION=3.10.0

FROM ubuntu:${UBUNTU_VERSION} AS arrayfire-installer
ARG AF_VERSION
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
RUN curl -fL --retry 5 -o /ArrayFire-v${AF_VERSION}_Linux_x86_64.sh \
        https://arrayfire.s3.amazonaws.com/${AF_VERSION}/ArrayFire-v${AF_VERSION}_Linux_x86_64.sh

# Unpack the installer's payload, leaving out the oneAPI backend (5.7 GB) and
# the debug symbols (3.6 GB), which cocos does not use.
FROM ubuntu:${UBUNTU_VERSION} AS arrayfire
ARG AF_VERSION
ARG AF_SHA256=f1289b0969d1c6db3bf52e961076f7b241764c80e5d509e363a122a219a82288
RUN --mount=type=bind,from=arrayfire-installer,target=/installer \
    set -eu; \
    installer=/installer/ArrayFire-v${AF_VERSION}_Linux_x86_64.sh; \
    echo "${AF_SHA256}  ${installer}" | sha256sum -c -; \
    offset=$(head -c 65536 "$installer" \
             | sed -n 's/^tail \$use_new_tail_syntax +\([0-9]*\) "\$0" | gunzip .*/\1/p'); \
    test "$offset" -gt 0; \
    mkdir -p /opt/arrayfire; \
    tail -n +"$offset" "$installer" | gunzip | tar -x -C /opt/arrayfire \
        --exclude='*.debug' --exclude='*oneapi*' --exclude='*sycl*' \
        --exclude='libur_*' --exclude='libumf*' --exclude='libintelocl*' \
        --exclude='share/ArrayFire/doc' --exclude='share/ArrayFire/examples'; \
    test -e /opt/arrayfire/lib64/libafcuda.so.3

FROM ubuntu:${UBUNTU_VERSION}
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

COPY --from=arrayfire /opt/arrayfire /opt/arrayfire
RUN echo /opt/arrayfire/lib64 > /etc/ld.so.conf.d/arrayfire.conf && ldconfig \
 && mkdir -p /etc/OpenCL/vendors \
 && echo libnvidia-opencl.so.1 > /etc/OpenCL/vendors/nvidia.icd
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /usr/local/bin/

ENV AF_PATH=/opt/arrayfire \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH

# Run as an unprivileged user with uid 1000, so that files written to a
# bind-mounted host directory belong to the usual first user on the host.
RUN userdel -r ubuntu \
 && useradd --create-home --uid 1000 --shell /bin/bash cocos \
 && install -d -o cocos -g cocos /opt/python /opt/venv
USER cocos
WORKDIR /home/cocos/cocos

# Dependencies first, so that source changes do not invalidate this layer.
COPY --chown=cocos:cocos .python-version pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/home/cocos/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project --group notebook
COPY --chown=cocos:cocos . .
RUN --mount=type=cache,target=/home/cocos/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --group notebook \
 && python -c "import cocos, arrayfire; print('cocos', cocos.__version__, '| ArrayFire', arrayfire.get_version())"

EXPOSE 8888
CMD ["bash"]
