#!/bin/bash
#
# Build MIGraphX prerequisites for docker container

# Used for onnx backend testing
pip3 install onnx==1.7.0 pytest

# rocblas and miopen
apt update && apt install -y rocblas miopen-hip

# pybind11
cd /src
git clone https://github.com/pybind/pybind11
pip3 install pytest
cd pybind11
git checkout d159a563383d10c821ba7b2a71905d1207db6de4
mkdir build
cd build
cmake ..
make -j4
make install

# protobuf
cd /src
git clone https://github.com/protocolbuffers/protobuf
cd protobuf
git checkout v3.11.0
git submodule update --init --recursive
./autogen.sh
./configure
make -j4
make install

# blaze
cd /src
wget https://bitbucket.org/blaze-lib/blaze/downloads/blaze-3.5.tar.gz
tar xf blaze-3.5.tar.gz
cd blaze-3.5
cp -r blaze /usr/local/include
cd ..
rm blaze-3.5.tar.gz

# half
# already copied by dockerfile

# json
cd /src
git clone https://github.com/nlohmann/json
cd json
git checkout v3.8.0
mkdir build
cd build
cmake ..
make
make install

# msgpack
apt install -y doxygen
cd /src
git clone https://github.com/msgpack/msgpack-c
cd msgpack-c
git checkout cpp-3.3.0
mkdir build
cd build
cmake -DMSGPACK_BUILD_TESTS=Off ..
make
make install

# OneDNN
apt install -y libomp-dev
cd /src
git clone https://github.com/oneapi-src/oneDNN.git
cd oneDNN
git checkout v1.7
mkdir build
cd build
env CXX=/opt/rocm/llvm/bin/clang++ cmake -DDNNL_CPU_RUNTIME=OMP ..
make
make install
