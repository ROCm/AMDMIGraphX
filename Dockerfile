FROM ubuntu:16.04

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

# Install cppcheck
RUN cget -p $PREFIX install danmar/cppcheck@ab02595be1b17035b534db655f9e119080a368bc

RUN cget -p $PREFIX install pfultz2/rocm-recipes

# Install dependencies
ADD requirements.txt /requirements.txt
RUN cget -p $PREFIX install -f /requirements.txt

# Install doc requirements
# ADD doc/requirements.txt /doc-requirements.txt
# RUN pip install -r /doc-requirements.txt
