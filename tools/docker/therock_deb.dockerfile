FROM ubuntu:24.04

ARG GPU_ARCH=gfx120x
ARG ROCM_VERSION=7.13
ARG ROCM_NIGHTLY_URL=https://rocm.nightlies.amd.com/deb/20260401-23832802691

# Support multiarch
RUN dpkg --add-architecture i386

# Install rocm key
RUN apt-get update && apt-get install -y software-properties-common gnupg2 --no-install-recommends curl && \
    curl -sL http://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -

# Add rocm repository
RUN sh -c 'echo deb [arch=amd64 trusted=yes] ${ROCM_NIGHTLY_URL} stable main > /etc/apt/sources.list.d/rocm.list'

# From docs.amd.com for installing rocm. Needed to install properly
RUN sh -c "echo 'Package: *\nPin: release o=repo.radeon.com\nPin-priority: 600' > /etc/apt/preferences.d/rocm-pin-600"

# Add LLVM repository for Clang 17 (ROCm 7.x ships with Clang 20 which has ODR false positives in ASAN)
RUN curl -sL https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    add-apt-repository -y "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-17 main"

# Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    apt-utils \
    bison \
    build-essential \
    clang-17 \
    cmake \
    curl \
    flex \
    g++ \
    gdb \
    git \
    lcov \
    locales \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-full \
    wget \
    libnuma-dev \
    libomp-17-dev \
    libssl-dev \
    zlib1g-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install rocm libraries
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    amdrocm-runtime \
    amdrocm-runtime-dev \
    amdrocm-llvm \
    amdrocm-core-${GPU_ARCH} \
    amdrocm-core-dev-${GPU_ARCH} \
    amdrocm-blas-${GPU_ARCH} \
    amdrocm-blas-dev-${GPU_ARCH} \
    amdrocm-hipblas-common-dev-${GPU_ARCH} \
    amdrocm-dnn-${GPU_ARCH} \
    amdrocm-dnn-dev-${GPU_ARCH} && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install rocm-cmake-build-tools from github
RUN git clone https://github.com/ROCm/rocm-cmake-build-tools.git && \
    cd rocm-cmake-build-tools && \
    cmake -B build -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build build --target install && \
    cd .. && rm -rf rocm-cmake-build-tools

# Install rbuild
RUN pip config set global.break-system-packages true
RUN pip install --break-system-packages https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz

# Add ROCm bin to PATH
ENV PATH=/opt/rocm/bin:$PATH

# Set locale
RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8