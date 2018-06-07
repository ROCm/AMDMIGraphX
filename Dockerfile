FROM ubuntu:xenial-20180417

ARG PREFIX=/usr/local

# Support multiarch
RUN dpkg --add-architecture i386

# Add rocm repository
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl apt-utils wget
RUN curl https://raw.githubusercontent.com/RadeonOpenCompute/ROCm-docker/develop/add-rocm.sh | bash

# Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    apt-utils \
    build-essential \
    clang-5.0 \
    clang-format-5.0 \
    clang-tidy-5.0 \
    cmake \
    curl \
    doxygen \
    git \
    hcc \
    hip_base \
    lcov \
    libnuma-dev \
    python \
    python-dev \
    python-pip \
    rocm-opencl \
    rocm-opencl-dev \
    software-properties-common \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install cget
RUN pip install cget

# Use hcc
RUN cget -p $PREFIX init --cxx /opt/rocm/bin/hcc

# Install dependencies
ADD dev-requirements.txt /dev-requirements.txt
ADD requirements.txt /requirements.txt
RUN cget -p $PREFIX install -f /dev-requirements.txt

# Install doc requirements
# ADD doc/requirements.txt /doc-requirements.txt
# RUN pip install -r /doc-requirements.txt
