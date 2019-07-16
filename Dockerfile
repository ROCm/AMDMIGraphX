FROM ubuntu:xenial-20180417

ARG PREFIX=/usr/local

# Support multiarch
RUN dpkg --add-architecture i386

# Add rocm repository
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl apt-utils wget software-properties-common
RUN curl https://raw.githubusercontent.com/RadeonOpenCompute/ROCm-docker/master/add-rocm.sh | bash

# Add ubuntu toolchain
RUN apt-get update && add-apt-repository ppa:ubuntu-toolchain-r/test -y

# Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    apt-utils \
    build-essential \
    clang-5.0 \
    clang-format-5.0 \
    clang-tidy-5.0 \
    cmake \
    comgr \
    curl \
    doxygen \
    g++-7 \
    gdb \
    git \
    hsa-rocr-dev \
    hsakmt-roct-dev \
    lcov \
    libelf-dev \
    libncurses5-dev \
    libnuma-dev \
    libpthread-stubs0-dev \
    python \
    python-dev \
    python-pip \
    rocm-device-libs \
    rocm-opencl \
    rocm-opencl-dev \
    rocminfo \
    software-properties-common \
    wget \
    zlib1g-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install cget
RUN pip install cget

# Install rclone
RUN pip install https://github.com/pfultz2/rclone/archive/master.tar.gz

# Install hcc
RUN rclone -b roc-2.6.x -c 0f4c96b7851af2663a7f3ac16ecfb76c7c78a5bf https://github.com/RadeonOpenCompute/hcc.git /hcc
RUN cget -p $PREFIX install hcc,/hcc

# Use hcc
RUN cget -p $PREFIX init --cxx $PREFIX/bin/hcc

# Workaround hip's broken cmake
RUN ln -s $PREFIX /opt/rocm/hip
RUN ln -s $PREFIX /opt/rocm/hcc

# Install dependencies
ADD dev-requirements.txt /dev-requirements.txt
ADD requirements.txt /requirements.txt
RUN cget -p $PREFIX install -f /dev-requirements.txt -DMIOPEN_CACHE_DIR=""

ENV MIOPEN_FIND_DB_PATH=/tmp/miopen/find-db
ENV MIOPEN_USER_DB_PATH=/tmp/miopen/user-db

ENV LD_LIBRARY_PATH=$PREFIX/lib

# Install doc requirements
ADD doc/requirements.txt /doc-requirements.txt
RUN pip install -r /doc-requirements.txt

# Setup ubsan environment to printstacktrace
RUN ln -s /usr/bin/llvm-symbolizer-5.0 /usr/local/bin/llvm-symbolizer
ENV UBSAN_OPTIONS=print_stacktrace=1
ENV ASAN_OPTIONS=detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1
