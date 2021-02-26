FROM ubuntu:18.04

ARG PREFIX=/usr/local

# Support multiarch
RUN dpkg --add-architecture i386

# Add rocm repository
RUN sh -c 'echo deb [arch=amd64 trusted=yes] http://repo.radeon.com/rocm/apt/.apt_3.7/ xenial main > /etc/apt/sources.list.d/rocm.list'

# Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    apt-utils \
    build-essential \
    clang-format-5.0 \
    cmake \
    curl \
    doxygen \
    g++-5 \
    g++-7 \
    gdb \
    git \
    lcov \
    locales \
    pkg-config \
    python \
    python-dev \
    python-pip \
    python3 \
    python3-dev \
    python3-pip \
    software-properties-common \
    wget \
    rocm-device-libs \
    miopen-hip \
    rocblas \
    zlib1g-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Workaround broken rocm packages
RUN ln -s /opt/rocm-* /opt/rocm
RUN echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf
RUN echo "/opt/rocm/llvm/lib" > /etc/ld.so.conf.d/rocm-llvm.conf
RUN ldconfig

RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install rbuild
RUN pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz

# Install yapf
RUN pip3 install yapf==0.28.0

# Install doc requirements
ADD doc/requirements.txt /doc-requirements.txt
RUN pip3 install -r /doc-requirements.txt

RUN pip3 install onnx==1.7.0 numpy==1.18.5 typing==3.7.4 pytest==6.0.1

# Download real models to run onnx unit tests
ENV ONNX_HOME=$HOME
COPY ./tools/download_models.sh /
RUN /download_models.sh && rm /download_models.sh

# Install dependencies
ADD dev-requirements.txt /dev-requirements.txt
ADD requirements.txt /requirements.txt
COPY ./tools/install_prereqs.sh /
RUN /install_prereqs.sh /usr/local / && rm /install_prereqs.sh

# Install latest ccache version
RUN cget -p $PREFIX install facebook/zstd@v1.4.5 -X subdir -DCMAKE_DIR=build/cmake
RUN cget -p $PREFIX install ccache@v4.1

# Install newer cmake for onnx runtime
RUN cget -p /opt/cmake install kitware/cmake@v3.13.0

ARG ONNXRUNTIME_REPO=https://github.com/Microsoft/onnxruntime
ARG ONNXRUNTIME_BRANCH=master
ARG ONNXRUNTIME_COMMIT=24f1bd6156cf5968bbc76dfb0e801a9b9c56b9fc
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

