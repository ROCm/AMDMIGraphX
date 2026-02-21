# Based on microsoft/onnxruntime dockerfiles/Dockerfile.migraphx
# https://github.com/microsoft/onnxruntime/blob/a98c9120db910537f83c7436deaa0fb42d8d57b6/dockerfiles/Dockerfile.migraphx
# --------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------
# Dockerfile to run ONNXRuntime with MIGraphX integration
#--------------------------------------------------------------------------

FROM rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

ARG ONNXRUNTIME_REPO=https://github.com/microsoft/onnxruntime
ARG ONNXRUNTIME_BRANCH=main

WORKDIR /

# Pin onnxruntime commit from AMDMIGraphX repo (used by Check ORT image tag)
COPY test/onnx/.onnxrt-commit /.onnxrt-commit

# Prepare onnxruntime repository at /onnxruntime for build_and_test_onnxrt.sh
RUN git clone --single-branch --branch ${ONNXRUNTIME_BRANCH} --recursive ${ONNXRUNTIME_REPO} onnxruntime && \
    cd onnxruntime && git checkout $(cat /.onnxrt-commit) && \
    /bin/sh /onnxruntime/dockerfiles/scripts/install_common_deps.sh

# Install half package and gdb required by the test stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    gdb \
    half \
    locales && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Add AMDMIGraphX CI test scripts (layout expected by build_and_test_onnxrt.sh)
ADD tools/build_and_test_onnxrt.sh /onnxruntime/build_and_test_onnxrt.sh
ADD tools/pai_test_launcher.sh /onnxruntime/tools/ci_build/github/pai/pai_test_launcher.sh
ADD tools/pai_provider_test_launcher.sh /onnxruntime/tools/ci_build/github/pai/pai_provider_test_launcher.sh
