#!/bin/bash
#
# Build MIGraphX prerequisites for docker container

#install pip3
sudo apt update && sudo apt install -y python3-pip

# install onnx package for unit tests
pip3 install onnx==1.7.0 numpy==1.18.5 typing==3.7.4 pytest==6.0.1

# install rbuild to build dependencies
pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz

# rocblas and miopen
sudo apt update && apt install -y rocblas miopen-hip 

PREFIX=/usr/local
REQ_FILE_DIR=""
if [ "$#" -ge 2 ]; then
  PREFIX=$1
  REQ_FILE_DIR=$2
elif [ "$#" -eq 1 ]; then
  PREFIX=$1
fi

echo "Dependencies are install at $PREFIX"

# Manually ignore rocm dependencies
cget -p $PREFIX ignore \
    RadeonOpenCompute/clang-ocl \
    ROCm-Developer-Tools/HIP \
    ROCmSoftwarePlatform/MIOpen \
    ROCmSoftwarePlatform/MIOpenGEMM \
    ROCmSoftwarePlatform/rocBLAS
cget -p $PREFIX init --cxx /opt/rocm/llvm/bin/clang++
cget -p $PREFIX install -f ${REQ_FILE_DIR}dev-requirements.txt
cget -p $PREFIX install oneapi-src/oneDNN@v1.7

# add the /usr/local/lib to link folder
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PREFIX/lib

