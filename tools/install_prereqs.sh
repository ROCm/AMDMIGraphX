#!/bin/bash

#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################
#
# Install the minimal set of packages required to build MIGraphX from source.
#
# The host distribution is detected automatically and the matching package
# manager (apt, dnf/yum, or zypper) is used. The ROCm components come from
# TheRock "amdrocm-*" packages, either from the system package manager (default)
# or from Python wheels when --whl is supplied.
#
# Usage:
#   install_build_prereqs.sh [--rocm-version <ver>] [--gpu <arch>] [--whl]
#
#   --rocm-version <ver>  ROCm release version used in versioned package names,
#                         e.g. 7.13 -> amdrocm-developer-tools7.13
#   --gpu <arch>          Specific GPU architecture, e.g. gfx942. When set, the
#                         smaller "skinny" per-gfx ROCm device code is installed.
#                         When unset, the "fat" (all-architecture) ROCm device
#                         code is installed, which is convenient for CI images
#                         that must run on any GPU.
#   --whl                 Install the ROCm "amdrocm-*" components from Python
#                         wheels (pip) instead of the system package manager.

set -eo pipefail

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PIP_BREAK_SYSTEM_PACKAGES=1

ROCM_VERSION=""
GPU_ARCH=""
USE_WHL=0

usage()
{
    grep '^#' "$0" | grep -v '!/bin/bash' | sed 's/^#//'
}

# require_value errors out when a flag that needs an argument is missing one.
require_value()
{
    if [[ $# -lt 2 ]]; then
        echo "Option $1 requires a value." >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rocm-version)
            require_value "$@"
            ROCM_VERSION="$2"
            shift 2
            ;;
        --rocm-version=*)
            ROCM_VERSION="${1#*=}"
            shift
            ;;
        --gpu)
            require_value "$@"
            GPU_ARCH="$2"
            shift 2
            ;;
        --gpu=*)
            GPU_ARCH="${1#*=}"
            shift
            ;;
        --whl)
            USE_WHL=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

# Detect the available package manager.
PKG_MGR=""
if command -v apt-get > /dev/null 2>&1; then
    PKG_MGR="apt"
elif command -v dnf > /dev/null 2>&1; then
    PKG_MGR="dnf"
elif command -v yum > /dev/null 2>&1; then
    PKG_MGR="yum"
elif command -v zypper > /dev/null 2>&1; then
    PKG_MGR="zypper"
else
    echo "No supported package manager found (apt, dnf, yum, zypper)." >&2
    exit 1
fi

echo "Using package manager: ${PKG_MGR}"

# pkg_install installs the given list of packages with the detected manager.
pkg_install()
{
    case "$PKG_MGR" in
        apt)
            DEBIAN_FRONTEND=noninteractive apt-get install -y "$@"
            ;;
        dnf)
            dnf install -y "$@"
            ;;
        yum)
            yum install -y "$@"
            ;;
        zypper)
            zypper -n --gpg-auto-import-keys install -y "$@"
            ;;
    esac
}

# pkg_refresh updates the package metadata for the detected manager.
pkg_refresh()
{
    case "$PKG_MGR" in
        apt) apt-get update ;;
        dnf) dnf makecache ;;
        yum) yum makecache ;;
        zypper) zypper -n --gpg-auto-import-keys refresh ;;
    esac
}

# Build prerequisites needed to compile MIGraphX, expressed per package manager
# since the package names differ across distributions.
case "$PKG_MGR" in
    apt)
        BUILD_PKGS=(
            apt-utils
            bison
            build-essential
            clang
            cmake
            curl
            flex
            g++
            gdb
            git
            lcov
            locales
            pkg-config
            python3
            python3-dev
            python3-pip
            python3-full
            wget
            libnuma-dev
            libomp-dev
            libtbb-dev
            libssl-dev
            zlib1g-dev
        )
        DEV_SUFFIX="-dev"
        ;;
    dnf | yum)
        BUILD_PKGS=(
            bison
            clang
            cmake
            curl
            flex
            gcc
            gcc-c++
            gdb
            git
            glibc-langpack-en
            lcov
            make
            pkgconfig
            python3
            python3-devel
            python3-pip
            wget
            numactl-devel
            libomp-devel
            tbb-devel
            openssl-devel
            zlib-devel
        )
        DEV_SUFFIX="-devel"
        ;;
    zypper)
        # The SLE15 base image only ships the limited SLE_BCI repo. clang, lcov
        # and libomp-devel are not available there and are not needed on SLES:
        # MIGraphX is built with the ROCm toolchain (/opt/rocm/llvm/bin/clang++)
        # and its OpenMP runtime, and coverage (lcov) is not run on SLES.
        BUILD_PKGS=(
            bison
            cmake
            curl
            flex
            gcc
            gcc-c++
            gdb
            git
            make
            pkg-config
            python3
            python3-devel
            python3-pip
            wget
            libnuma-devel
            libopenssl-devel
            zlib-devel
        )
        DEV_SUFFIX="-devel"
        ;;
esac

pkg_refresh
pkg_install "${BUILD_PKGS[@]}"

# Minimal set of ROCm components required to configure, compile, link, and
# package MIGraphX. Only the libraries MIGraphX actually resolves via cmake
# find_package are pulled in (HIP/HSA/hiprtc, rocBLAS/hipBLAS/hipBLASLt, MIOpen,
# hipBLAS-common); composable_kernel and rocMLIR are built from source.
#
# Mapping of MIGraphX requirement -> owning amdrocm package:
#   compiler (clang++/hipcc) + device bitcode  amdrocm-llvm      (pulled by runtime)
#   rocm_version.h, rocminfo                    amdrocm-base      (pulled by runtime)
#   hip / hsa-runtime64 / hiprtc                amdrocm-runtime(-dev)
#   rocblas / hipblas / hipblaslt               amdrocm-blas(-dev)
#   miopen                                      amdrocm-dnn(-dev)
#   hipblas-common headers                      amdrocm-hipblas-common-dev
#
# amdrocm-runtime-dev transitively installs amdrocm-runtime, amdrocm-base,
# amdrocm-llvm and amdrocm-llvm-dev, so the whole compiler/HIP stack comes with
# it. The -dev packages for blas/dnn do NOT pull their runtime libraries, so
# those are listed explicitly. Per-gfx device-code packages are NOT required to
# build or package; they are only needed to run on a GPU and are added below
# when a target architecture is supplied.
if [[ "$USE_WHL" -eq 1 ]]; then
    # Install the ROCm components from Python wheels.
    pip3 install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "rocm[libraries,devel]" --target /opt/rocm

else
    ROCM_PKGS=(
        "amdrocm-runtime${DEV_SUFFIX}${ROCM_VERSION}"
        "amdrocm-blas${DEV_SUFFIX}${ROCM_VERSION}"
        "amdrocm-blas${ROCM_VERSION}"
        "amdrocm-dnn${DEV_SUFFIX}${ROCM_VERSION}"
        "amdrocm-dnn${ROCM_VERSION}"
        "amdrocm-hipblas-common${DEV_SUFFIX}${ROCM_VERSION}"
    )

    # Device code is optional for the build itself (the host libraries above are
    # enough to configure/compile/link/package), but is required to run on a GPU.
    # The target architecture is taken solely from --gpu; there is no rocminfo
    # auto-detection:
    #   * --gpu <arch> set -> add the small "skinny" per-gfx packages.
    #     amdrocm-dnn<ver>-<arch> pulls the matching rocBLAS and solver device
    #     code transitively.
    #   * --gpu unset      -> add the "fat" amdrocm-core<ver> umbrella, which
    #     carries device code for every supported architecture so the resulting
    #     build runs on any GPU (convenient for CI).
    if [[ -n "$GPU_ARCH" ]]; then
        echo "Adding skinny device-code packages for ${GPU_ARCH}"
        ROCM_PKGS+=(
            "amdrocm-blas${ROCM_VERSION}-${GPU_ARCH}"
            "amdrocm-dnn${ROCM_VERSION}-${GPU_ARCH}"
        )
    else
        echo "No GPU architecture specified or detected; installing fat (all-architecture) ROCm device code"
        ROCM_PKGS+=("amdrocm-core${ROCM_VERSION}")
    fi
    pkg_install "${ROCM_PKGS[@]}"
fi

pip install setuptools wheel

# rbuild is used to build the MIGraphX third-party dependencies. This pip
# install always runs, independent of the --whl flag.
pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz
