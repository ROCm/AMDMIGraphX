FROM ubuntu:18.04

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
    g++-5 \
    g++-7 \
    gdb \
    git \
    hsa-rocr-dev \
    hsakmt-roct-dev \
    lcov \
    libelf-dev \
    libfile-which-perl \
    libncurses5-dev \
    libnuma-dev \
    libpthread-stubs0-dev \
    libssl-dev \
    locales \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python-pip \
    python-dev \
    rocm-device-libs \
    rocm-opencl \
    rocm-opencl-dev \
    software-properties-common \
    sudo \
    wget \
    zlib1g-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Wokaround broken rocm packages for rocm >= 3.1 
RUN [ -d /opt/rocm ] || ln -sd $(realpath /opt/rocm-*) /opt/rocm

RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install cget
RUN pip3 install cget 

# Install rclone
RUN pip install https://github.com/pfultz2/rclone/archive/master.tar.gz

# Install yapf
RUN pip3 install yapf==0.28.0

# Install doc requirements
ADD doc/requirements.txt /doc-requirements.txt
RUN pip3 install -r /doc-requirements.txt

# Install hcc
RUN rclone -b roc-3.0.x -c 286651a04d9c3a8e3052dd84b1822985498cd27d https://github.com/RadeonOpenCompute/hcc.git /hcc
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

RUN pip3 install onnx==1.7.0 numpy==1.18.5 typing==3.7.4 pytest==6.0.1

# Download real models to run onnx unit tests
ENV ONNX_HOME=$HOME
COPY ./tools/download_models.sh /
RUN chmod +x /download_models.sh && /download_models.sh && rm /download_models.sh

# Install newer cmake for onnx runtime
RUN cget -p /opt/cmake install kitware/cmake@v3.13.0

ARG ONNXRUNTIME_REPO=https://github.com/Microsoft/onnxruntime
ARG ONNXRUNTIME_BRANCH=master
ARG ONNXRUNTIME_COMMIT=417929b049829c44bcd59c0d0eae7ae6c71ab111
RUN git clone --single-branch --branch ${ONNXRUNTIME_BRANCH} --recursive ${ONNXRUNTIME_REPO} onnxruntime && \
    cd onnxruntime && \
    git checkout ${ONNXRUNTIME_COMMIT} && \
    /bin/sh dockerfiles/scripts/install_common_deps.sh

ADD tools/build_and_test_onnxrt.sh /onnxruntime/build_and_test_onnxrt.sh

ENV MIOPEN_FIND_DB_PATH=/tmp/miopen/find-db
ENV MIOPEN_USER_DB_PATH=/tmp/miopen/user-db

ENV LD_LIBRARY_PATH=$PREFIX/lib

# Setup ubsan environment to printstacktrace
ENV UBSAN_OPTIONS=print_stacktrace=1
ENV ASAN_OPTIONS=detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1
