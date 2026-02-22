FROM ubuntu:22.04

# Install rocm key
RUN apt-get update && apt-get install -y software-properties-common gnupg2 --no-install-recommends curl && \
    curl -sL http://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -

# Add rocm repository
RUN sh -c 'echo deb [arch=amd64 trusted=yes] http://repo.radeon.com/rocm/apt/7.1.1/ jammy main > /etc/apt/sources.list.d/rocm.list'

# From docs.amd.com for installing rocm. Needed to install properly
RUN sh -c "echo 'Package: *\nPin: release o=repo.radeon.com\nPin-priority: 600' > /etc/apt/preferences.d/rocm-pin-600"


ARG ONNXRUNTIME_REPO=https://github.com/microsoft/onnxruntime
ARG ONNXRUNTIME_BRANCH=main

WORKDIR /

# Pin onnxruntime commit from AMDMIGraphX repo (used by Check ORT image tag)
COPY test/onnx/.onnxrt-commit /.onnxrt-commit

# Install half package and gdb required by the test stage
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    cmake \
    gdb \
    git \
    half \
    locales \
    pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Workaround broken rocm packages
RUN ln -s /opt/rocm-* /opt/rocm
RUN echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf
RUN echo "/opt/rocm/llvm/lib" > /etc/ld.so.conf.d/rocm-llvm.conf
RUN ldconfig

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

# Install pytorch
RUN pip3 install https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/torch-2.8.0%2Brocm7.1.1.lw.gitcba8b9d2-cp310-cp310-linux_x86_64.whl\
                 https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/torchvision-0.24.0%2Brocm7.1.1.gitb919bd0c-cp310-cp310-linux_x86_64.whl\
                 https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/triton-3.4.0%2Brocm7.1.1.git0cace8d2-cp310-cp310-linux_x86_64.whl

RUN pip install cmake==3.28
