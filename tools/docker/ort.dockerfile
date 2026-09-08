FROM ubuntu:24.04

ARG ROCM_VERSION=10.0
ARG GPU_ARCH=""

ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Install rocm key
RUN apt-get update && apt-get install -y software-properties-common gnupg2 --no-install-recommends curl && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://stable.repo.amd.com/rocm/gpg/packages.gpg | gpg --dearmor -o /etc/apt/keyrings/amdrocm.gpg

# Add rocm repository
RUN sh -c 'echo deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] https://stable.repo.amd.com/rocm/core/packages/ubuntu2404 stable main > /etc/apt/sources.list.d/rocm.list'

ARG ONNXRUNTIME_REPO=https://github.com/microsoft/onnxruntime
ARG ONNXRUNTIME_BRANCH=main

WORKDIR /

# Pin onnxruntime commit from AMDMIGraphX repo (used by Check ORT image tag)
COPY test/onnx/.onnxrt-commit /.onnxrt-commit

# Install gdb required by the test stage
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    gdb \
    git \
    libsqlite3-dev \
    locales \
    python3 \
    python3-dev \
    python3-pip \
    python3-full \
    pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install pipx

RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY ./tools/install_prereqs.sh /
COPY ./tools/requirements-py.txt /requirements-py.txt
RUN ./install_prereqs.sh \
        --rocm-only \
        --rocm-version ${ROCM_VERSION} \
        ${GPU_ARCH:+--gpu ${GPU_ARCH}} \
        ${USE_WHL:+--whl}
RUN rm /install_prereqs.sh && rm /*.txt

# Workaround broken rocm packages
RUN echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf
RUN echo "/opt/rocm/llvm/lib" > /etc/ld.so.conf.d/rocm-llvm.conf
RUN ldconfig

# Prepare onnxruntime repository at /onnxruntime for build_and_test_onnxrt.sh
RUN git clone --single-branch --branch ${ONNXRUNTIME_BRANCH} --recursive ${ONNXRUNTIME_REPO} onnxruntime && \
    cd onnxruntime && git checkout $(cat /.onnxrt-commit) && \
    /bin/sh /onnxruntime/dockerfiles/scripts/install_common_deps.sh

# Add AMDMIGraphX CI test scripts (layout expected by build_and_test_onnxrt.sh)
ADD tools/build_and_test_onnxrt.sh /onnxruntime/build_and_test_onnxrt.sh
ADD tools/pai_test_launcher.sh /onnxruntime/tools/ci_build/github/pai/pai_test_launcher.sh
ADD tools/pai_provider_test_launcher.sh /onnxruntime/tools/ci_build/github/pai/pai_provider_test_launcher.sh

RUN pipx install --global cmake==4.3.1
