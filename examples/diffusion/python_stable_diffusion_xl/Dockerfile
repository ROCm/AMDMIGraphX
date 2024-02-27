# build with:
# GPU_ARCH="$(/opt/rocm/bin/rocminfo | grep -o -m 1 'gfx.*')"
# docker build -t sdxl_perf --build-arg GPU_ARCH="$GPU_ARCH" .
FROM rocm/pytorch-private:rocm13360_445_ubuntu22.04_py3.9_pytorch_tunablegemm-temp_aed2c82_with_apex_v2

ARG ROCM_PATH=/opt/rocm
ARG GPU_ARCH

# install rbuild
RUN pip install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz

# update rocm-cmake
RUN git clone https://github.com/RadeonOpenCompute/rocm-cmake.git \
    && cd rocm-cmake  \
    && git checkout 5a34e72d9f113eb5d028e740c2def1f944619595 \
    && mkdir build && cd build \
    && cmake .. && cmake --build . --target install

# install migraphx
RUN git clone --single-branch --branch sdxl_perf_2-27 --recursive https://github.com/ROCm/AMDMIGraphX.git \
    && cd AMDMIGraphX \
    && rbuild build -d depend -DBUILD_TESTING=Off -DCMAKE_INSTALL_PREFIX=/opt/rocm/ --cxx=/opt/rocm/llvm/bin/clang++ -DGPU_TARGETS=$GPU_ARCH \
    && cd build && make install


# install torch_migraphx and stable diffusion deps
RUN pip3 install pybind11-global diffusers transformers accelerate

# install torch_migraphx
RUN git clone https://github.com/ROCm/torch_migraphx.git \
    && cd torch_migraphx/py \
    && export TORCH_CMAKE_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") \
    && python -m pip install .


WORKDIR /workspace
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib
ENV PYTHONPATH=${ROCM_PATH}/lib
ENTRYPOINT [ "/bin/bash" ]