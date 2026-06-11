# MIGraphX build environment with TheRock (amdrocm-*) deb packages.
#
# Build args:
#   ROCM_VERSION      ROCm release version for versioned package names (e.g. 7.13)
#   GPU_ARCH          GPU architecture family (e.g. gfx120x, gfx94x)
#   USE_WHL           Set to a non-empty value to install ROCm from Python
#                     wheels (pip) instead of system packages
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
#   rbuild build -d depend -B build \
#       -DGPU_TARGETS=$(rocminfo | grep -o -m1 'gfx.*')

FROM ubuntu:24.04

# Fail a piped command if any stage fails (required for the key-dearmor pipe below).
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ROCm release version (used in versioned package names, e.g. amdrocm-developer-tools7.13)
ARG ROCM_VERSION="7.13"
# GPU architecture family (e.g. gfx942, gfx120x); leave empty for arch-independent packages
ARG GPU_ARCH=""

# Install prerequisites needed to fetch and dearmor the ROCm signing key.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates gnupg2 curl && \
    rm -rf /var/lib/apt/lists/*

# Register the ROCm apt repository and its signing key.
RUN mkdir --parents --mode=0755 /etc/apt/keyrings && \
    curl -fsSL https://repo.amd.com/rocm/packages/gpg/rocm.gpg | \
        gpg --dearmor | tee /etc/apt/keyrings/amdrocm.gpg > /dev/null && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] https://repo.amd.com/rocm/packages-multi-arch/ubuntu2404 stable main" \
        > /etc/apt/sources.list.d/rocm.list

# Install the MIGraphX build prerequisites (system packages + ROCm components).
# Set USE_WHL to any non-empty value to install ROCm from Python wheels instead
# of system packages (passes --whl to the prereqs script).
ARG USE_WHL=""
ENV PIP_BREAK_SYSTEM_PACKAGES=1

COPY tools/install_prereqs.sh /tmp/install_prereqs.sh
RUN chmod +x /tmp/install_prereqs.sh && \
    /tmp/install_prereqs.sh \
        --rocm-version ${ROCM_VERSION} \
        ${GPU_ARCH:+--gpu ${GPU_ARCH}} \
        ${USE_WHL:+--whl} && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Additional packages
RUN python3 -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch \
    "torch==2.11.0+rocm${ROCM_VERSION}.0" \
    "torchvision==0.26.0+rocm${ROCM_VERSION}.0" \
    "torchaudio==2.11.0+rocm${ROCM_VERSION}.0"

RUN python3 -m pip install onnxruntime clang-format==22.1.5

# Set locale
RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8

ENV LC_ALL=C.UTF-8

# Setup ubsan environment to printstacktrace
ENV UBSAN_OPTIONS=print_stacktrace=1
# Disable odr detection since its broken with shared libraries
# See: https://github.com/google/sanitizers/issues/1017
ENV ASAN_OPTIONS=detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1
RUN ln -s /opt/rocm/llvm/bin/llvm-symbolizer /usr/bin/llvm-symbolizer

