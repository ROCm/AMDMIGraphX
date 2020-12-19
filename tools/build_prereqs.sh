#!/bin/bash
#
# Build MIGraphX prerequisites for docker container

apt install -y python3-pip

# Used for onnx backend testing
pip3 install onnx==1.7.0 pytest

# rocblas and miopen
apt update && apt install -y rocblas miopen-hip 

cur_dir=`pwd`
deps=$cur_dir/depend_tmp

mkdir depend_tmp

# pybind11
cd $deps
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
cd $deps
apt install -y dh-autoreconf libtools
git clone https://github.com/protocolbuffers/protobuf
cd protobuf
git checkout v3.11.0
git submodule update --init --recursive
./autogen.sh
./configure
make -j4
make install

# blaze
apt install -y wget
cd $deps
wget https://bitbucket.org/blaze-lib/blaze/downloads/blaze-3.5.tar.gz
tar xf blaze-3.5.tar.gz
cd blaze-3.5
cp -r blaze /usr/local/include
cd ..
rm blaze-3.5.tar.gz

# half
cd $deps
wget https://github.com/pfultz2/half/archive/1.12.0.tar.gz
tar xf 1.12.0.tar.gz
cp half-1.12.0/include/half.hpp /usr/local/include/half.hpp

# json
cd $deps
git clone https://github.com/nlohmann/json
cd json
git checkout v3.8.0
mkdir build
cd build
cmake ..
make -j$(nproc)
make install

# msgpack
cd $deps
apt install -y doxygen
git clone https://github.com/msgpack/msgpack-c
cd msgpack-c
git checkout cpp-3.3.0
mkdir build
cd build
cmake -DMSGPACK_BUILD_TESTS=Off ..
make -j$(nproc)
make install

# OneDNN
cd $deps
apt install -y libomp-dev
git clone https://github.com/oneapi-src/oneDNN.git
cd oneDNN
git checkout v1.7
mkdir build
cd build
env CXX=/opt/rocm/llvm/bin/clang++ cmake -DDNNL_CPU_RUNTIME=OMP ..
make -j$(nproc)
make install

# add the /usr/local/lib to link folder
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

# remove the temp depend folder
rm -rf depend_tmp

