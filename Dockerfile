# MIGraphX build environment with TheRock (amdrocm-*) deb packages.
#
# Build args:
#   ROCM_VERSION      ROCm release version for versioned package names (e.g. 7.13)
#   GPU_ARCH          GPU architecture family (e.g. gfx120x, gfx94x)
#   USE_WHL           Set to a non-empty value to install ROCm from Python
#                     wheels (pip) instead of system packages
#   INDEX_URL         pip --index-url used when installing ROCm from wheels
#                     (only effective together with USE_WHL)
#
# Build:
#   docker build --build-arg GPU_ARCH=<gpu_arch> \
#       --build-arg ROCM_VERSION=<rocm_version> \
#       -t migraphx-therock .
#
# Run:
#   docker run -it --device=/dev/kfd --device=/dev/dri --group-add video \
#       -v $(pwd):/code/AMDMIGraphX migraphx-therock
#
# Build MIGraphX inside the container:
#   cd /code/AMDMIGraphX
#   rbuild build -d depend -B build
#

FROM ubuntu:24.04

ARG PREFIX=/usr/local
# ROCm release version (used in versioned package names, e.g. amdrocm-developer-tools7.13)
ARG ROCM_VERSION="7.14"
# GPU architecture family (e.g. gfx942, gfx120x); leave empty for arch-independent packages
ARG GPU_ARCH=""
# Install location for the prebuilt MIGraphX dependencies.
ARG PREFIX=/usr/local
# Install the MIGraphX build prerequisites (system packages + ROCm components).
# Set USE_WHL to any non-empty value to install ROCm from Python wheels instead
# of system packages (passes --whl to the prereqs script).
ARG USE_WHL=""
# pip index URL for the wheel-based ROCm install (only used when USE_WHL is set).
ARG INDEX_URL="https://repo.amd.com/rocm/whl-multi-arch/"

# Support multiarch
RUN dpkg --add-architecture i386

# Install rocm key
RUN apt-get update && apt-get install -y software-properties-common gnupg2 --no-install-recommends curl && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://repo.amd.com/rocm/packages/gpg/rocm.gpg | gpg --dearmor -o /etc/apt/keyrings/amdrocm.gpg

# Add rocm repository
RUN sh -c 'echo deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] https://repo.amd.com/rocm/packages-multi-arch/ubuntu2404 stable main > /etc/apt/sources.list.d/rocm.list'

# Add LLVM repository for Clang 17 (ROCm 7.x ships with Clang 20 which has ODR false positives in ASAN)
RUN curl -sL https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    add-apt-repository -y "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-17 main"

# Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
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
    libpython3.8 \
    unzip \
    wget \
    libnuma-dev \
    libomp-17-dev \
    libssl-dev \
    zlib1g-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Install dependencies
ADD dev-requirements.txt /dev-requirements.txt
ADD requirements.txt /requirements.txt
ADD rbuild.ini /rbuild.ini

COPY ./tools/install_prereqs.sh /
COPY ./tools/requirements-py.txt /requirements-py.txt
RUN ./install_prereqs.sh \
        --rocm-version ${ROCM_VERSION} \
        ${GPU_ARCH:+--gpu ${GPU_ARCH}} \
        --index-url ${INDEX_URL} \
        ${USE_WHL:+--whl}
RUN rm /install_prereqs.sh && rm /*.txt
RUN test -f /usr/local/hash || exit 1

# Workaround broken rocm packages
RUN echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf
RUN echo "/opt/rocm/llvm/lib" > /etc/ld.so.conf.d/rocm-llvm.conf
RUN ldconfig

# Manually remove rocm-cmake, since it shouldnt be installed in the first place
RUN rm -rf /opt/rocm/share/rocmcmakebuildtools

# Workaround broken rocm packages that are missing clangd, so install the
# upstream LLVM snapshot built from the same clang 23 base as the ROCm 7.14
# compiler, may still be a mismatch
ARG CLANGD_SNAPSHOT="20260621"
RUN curl -fsSL -o /tmp/clangd.zip \
        "https://github.com/clangd/clangd/releases/download/snapshot_${CLANGD_SNAPSHOT}/clangd-linux-snapshot_${CLANGD_SNAPSHOT}.zip" && \
    unzip -q /tmp/clangd.zip -d /opt && \
    mv "/opt/clangd_snapshot_${CLANGD_SNAPSHOT}" /opt/clangd && \
    chmod -R a+rX /opt/clangd && \
    ln -s /opt/clangd/bin/clangd /opt/rocm/llvm/bin/clangd && \
    ln -s /opt/clangd/bin/clangd /usr/local/bin/clangd && \
    rm /tmp/clangd.zip

# Install pytorch
RUN pip3 install --index-url "${INDEX_URL}" \
        "torch==2.11.0+rocm${ROCM_VERSION}.0" \
        "torchvision==0.26.0+rocm${ROCM_VERSION}.0" \
        "torchaudio==2.11.0+rocm${ROCM_VERSION}.0"

# Location where onnx unit tests models are cached
ENV ONNX_HOME=/.onnx
RUN mkdir -p $ONNX_HOME/models && chmod 777 $ONNX_HOME/models

# Install yapf
RUN pipx install --global yapf==0.28.0

# Install clang format
RUN pipx install --global clang-format==22.1.5

# Install doc requirements
ADD docs/sphinx/requirements.txt /doc-requirements.txt
# pip rejects extras in a constraints file (the pip-compile output pins
# pyjwt[crypto]), and extras carry no meaning in a constraint, so strip them to
# pin the sphinx install.
RUN sed 's/\[[^][]*\]//' /doc-requirements.txt > /doc-constraints.txt && \
    pipx install --global sphinx --pip-args="-c /doc-constraints.txt" && \
    rm /doc-constraints.txt
RUN pipx inject --global sphinx -r /doc-requirements.txt

# Install latest ccache version
RUN cget -p $PREFIX install facebook/zstd@v1.4.5 -X subdir -DCMAKE_DIR=build/cmake
RUN cget -p $PREFIX install ccache@v4.1 -DENABLE_TESTING=OFF
# Install a newer version of doxygen because the one that comes with ubuntu is broken
RUN cget -p $PREFIX install doxygen@Release_1_14_0

ENV MIOPEN_FIND_DB_PATH=/tmp/miopen/find-db
ENV MIOPEN_USER_DB_PATH=/tmp/miopen/user-db
ENV LD_LIBRARY_PATH=$PREFIX/lib

# Setup ubsan environment to printstacktrace
ENV UBSAN_OPTIONS=print_stacktrace=1
# Disable odr detection since its broken with shared libraries
# See: https://github.com/google/sanitizers/issues/1017
ENV ASAN_OPTIONS=detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1
RUN ln -s /opt/rocm/llvm/bin/llvm-symbolizer /usr/bin/llvm-symbolizer

