FROM ubuntu:22.04

ARG ONNXRUNTIME_REPO=https://github.com/microsoft/onnxruntime
ARG ONNXRUNTIME_BRANCH=main

WORKDIR /

# Install rocm key
RUN apt-get update && apt-get install -y software-properties-common gnupg2 --no-install-recommends curl && \
    curl -sL http://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -

# Add rocm repository
RUN sh -c 'echo deb [arch=amd64 trusted=yes] http://repo.radeon.com/rocm/apt/7.1.1/ jammy main > /etc/apt/sources.list.d/rocm.list'


# Pin onnxruntime commit from AMDMIGraphX repo (used by Check ORT image tag)
COPY test/onnx/.onnxrt-commit /.onnxrt-commit

# Install half package and gdb required by the test stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    gdb \
    git \
    half \
    locales \
    pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Prepare onnxruntime repository at /onnxruntime for build_and_test_onnxrt.sh
RUN git clone --single-branch --branch ${ONNXRUNTIME_BRANCH} --recursive ${ONNXRUNTIME_REPO} onnxruntime && \
    cd onnxruntime && git checkout $(cat /.onnxrt-commit) && \
    /bin/sh /onnxruntime/dockerfiles/scripts/install_common_deps.sh

RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Add AMDMIGraphX CI test scripts (layout expected by build_and_test_onnxrt.sh)
ADD tools/build_and_test_onnxrt.sh /onnxruntime/build_and_test_onnxrt.sh
ADD tools/pai_test_launcher.sh /onnxruntime/tools/ci_build/github/pai/pai_test_launcher.sh
ADD tools/pai_provider_test_launcher.sh /onnxruntime/tools/ci_build/github/pai/pai_provider_test_launcher.sh

RUN pip install torch torch-vision cmake==3.28
