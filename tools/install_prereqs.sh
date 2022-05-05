#!/bin/bash
#
# Build MIGraphX prerequisites for docker container

set -e

export LC_ALL=C.UTF-8
export LANG=C.UTF-8


# Need pip3 and Python headers to build dependencies
apt update && apt install -y python3-pip python3-dev cmake

# Needed for cmake to build various pip packages
pip3 install setuptools wheel

#install rocm-cmake, rocblas and miopen
apt install -y rocm-cmake rocblas miopen-hip openmp-extras

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

echo "Dependencies are install at $PREFIX"

# Install deps with rbuild
rbuild prepare -d $PREFIX

# install onnx package for unit tests
pip3 install onnx==1.8.1 numpy==1.18.5 typing==3.7.4 pytest==6.0.1 packaging==16.8

