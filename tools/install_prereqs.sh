#!/bin/bash
#
# Build MIGraphX prerequisites for docker container

set -e

export LC_ALL=C.UTF-8
export LANG=C.UTF-8


# Need pip3 and Python headers to build dependencies
apt update && apt install -y python3-pip python3-dev cmake rocm-cmake rocblas miopen-hip openmp-extras

# Needed for cmake to build various pip packages
pip3 install setuptools wheel

# install rbuild to build dependencies
pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz


PREFIX=/usr/local
REQ_FILE_DIR=""
if [ "$#" -ge 2 ]; then
  PREFIX=$1
  cd $2
elif [ "$#" -eq 1 ]; then
  PREFIX=$1
fi

echo "Dependencies are installed at $PREFIX"

# Install deps with rbuild
rbuild prepare -d $PREFIX -s develop

# install onnx package for unit tests
pip3 install onnx==1.8.1 numpy==1.18.5 typing==3.7.4 pytest==6.0.1 packaging==16.8

# pin version of protobuf in Python for onnx runtime unit tests
pip3 install protobuf==3.20.0
