FROM rocm/dev-ubuntu-22.04:6.2

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --allow-unauthenticated \
    apt-utils \
    cmake \
    migraphx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /mgx_llama2

COPY . /mgx_llama2

RUN rm -rf /mgx_llama2/build && mkdir /mgx_llama2/build

RUN cd /mgx_llama2/build && CXX=/opt/rocm/llvm/bin/clang++ cmake .. && make