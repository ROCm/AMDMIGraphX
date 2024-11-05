FROM rocm/dev-ubuntu-22.04:6.2

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --allow-unauthenticated \
    apt-utils \
    cmake \
    git  && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /migraphx && cd /migraphx && git clone --branch develop https://github.com/ROCm/AMDMIGraphX.git

WORKDIR /migraphx/AMDMIGraphX

RUN ./tools/install_prereqs.sh

RUN export test=$(/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*')

#TODO: use $(/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*') for GPU_TARGETS
RUN mkdir build && cd build && \
    CXX=/opt/rocm/llvm/bin/clang++ cmake .. -DGPU_TARGETS='gfx942' && \
    make -j$(nproc) && \
    make install

RUN mkdir /mgx_llama2

COPY . /mgx_llama2

RUN rm -rf /mgx_llama2/build && mkdir /mgx_llama2/build

WORKDIR /mgx_llama2/build

RUN CXX=/opt/rocm/llvm/bin/clang++ cmake .. && make