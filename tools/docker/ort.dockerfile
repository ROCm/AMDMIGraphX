FROM ubuntu:24.04

ARG ROCM_VERSION=7.14
ARG GPU_ARCH=""

ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Install prerequisites needed to fetch and dearmor the ROCm signing key.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gnupg2 \
    curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Register the ROCm apt repository and its signing key.
RUN mkdir --parents --mode=0755 /etc/apt/keyrings && \
    curl -fsSL https://repo.amd.com/rocm/packages/gpg/rocm.gpg | \
        gpg --dearmor | tee /etc/apt/keyrings/amdrocm.gpg > /dev/null && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] https://repo.amd.com/rocm/packages-multi-arch/ubuntu2404 stable main" \
        > /etc/apt/sources.list.d/rocm.list


ARG ONNXRUNTIME_REPO=https://github.com/microsoft/onnxruntime
ARG ONNXRUNTIME_BRANCH=main

WORKDIR /

# Pin onnxruntime commit from AMDMIGraphX repo (used by Check ORT image tag)
COPY test/onnx/.onnxrt-commit /.onnxrt-commit

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    gdb \
    git \
    locales \
    python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY tools/install_prereqs.sh /tmp/install_prereqs.sh
RUN chmod +x /tmp/install_prereqs.sh && \
    /tmp/install_prereqs.sh \
        --rocm-version ${ROCM_VERSION} \
        ${GPU_ARCH:+--gpu ${GPU_ARCH}} && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

 # TheRock installs into a versioned root (/opt/rocm/core-<ver>). Expose the
 # conventional /opt/rocm/{bin,lib,llvm,...} layout expected by MIGraphX tooling.
 RUN mkdir -p /opt/rocm && \
     for d in bin lib libexec include share llvm amdgcn; do \
         ln -snf core-${ROCM_VERSION}/$d /opt/rocm/$d; \
     done && \
     echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf && \
     echo "/opt/rocm/llvm/lib" > /etc/ld.so.conf.d/rocm-llvm.conf && \
     ldconfig

 ENV ROCM_PATH=/opt/rocm
 ENV PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH

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

RUN python3 -m pip install cmake==4.3.1
