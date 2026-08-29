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
#   install_prereqs.sh [--rocm-version <ver>] [--gpu <arch>] [--whl] [--index-url <url>]
#                      [--rocm-only]
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
#   --index-url <url>     Override the pip --index-url used with --whl. Common
#                         values to use would be
#                         https://repo.amd.com/rocm/whl-multi-arch
#                         https://rocm.prereleases.amd.com/whl-multi-arch
#                         https://rocm.nightlies.amd.com/whl-multi-arch/
#   --rocm-only           Install only the ROCm components and skip everything
#                         else (pipx, rbuild, and the MIGraphX dependencies).

set -eo pipefail

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

ROCM_VERSION=""
GPU_ARCH=""
USE_WHL=0
ROCM_ONLY=0
INDEX_URL="https://repo.amd.com/rocm/whl-multi-arch/"
PREFIX=/usr/local
REQ_FILE_DIR="$(dirname -- "$0")"

source /etc/os-release

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
        --rocm-only)
            ROCM_ONLY=1
            shift
            ;;
        --index-url)
            require_value "$@"
            INDEX_URL="$2"
            shift 2
            ;;
        --index-url=*)
            INDEX_URL="${1#*=}"
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

case "$PKG_MGR" in
    apt)
        DEV_SUFFIX="-dev"
        ;;
    dnf | yum | zypper)
        DEV_SUFFIX="-devel"
        ;;
esac

# Install the ROCm packages, by WHL or by OS package
if [[ "$USE_WHL" -eq 1 ]]; then
    # Install the ROCm components from Python wheels.
    echo "Using pip index URL: ${INDEX_URL}"
    python3 -m pip install --no-build-isolation --index-url "$INDEX_URL" \
        "rocm[libraries,devel,device-${GPU_ARCH:-all}]"
    rocm-sdk init

else
    pkg_refresh
    ROCM_PKGS=(
        "amdrocm-runtime${DEV_SUFFIX}${ROCM_VERSION}"
        "amdrocm-blas${DEV_SUFFIX}${ROCM_VERSION}"
        "amdrocm-blas-host${ROCM_VERSION}"
        "amdrocm-dnn${DEV_SUFFIX}${ROCM_VERSION}"
        "amdrocm-dnn-host${ROCM_VERSION}"
        "amdrocm-hipblas-common${DEV_SUFFIX}${ROCM_VERSION}"
    )

    #  --gpu <arch> set -> add the small "skinny" per-gfx packages.
    #  amdrocm-dnn<ver>-<arch> pulls the matching rocBLAS and solver device
    #  code transitively.
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
    mkdir -p /opt/rocm
    for d in bin lib libexec include share llvm amdgcn; do
        ln -snf core-${ROCM_VERSION}/$d /opt/rocm/$d;
    done
fi

if [[ "$ROCM_ONLY" -eq 1 ]]; then
    echo "ROCm is installed; skipping the remaining MIGraphX prerequisites"
    exit 0
fi

# Install pipx from pip rather than the distro package. The distro pipx depends
# on python3-packaging, and a dpkg-owned copy of packaging cannot be uninstalled
# by pip when requirements-py.txt pins a different version.
pip3 install setuptools wheel pipx pybind11

# Install apps to the global location so any user can run them. On SLES pipx
# runs under python 3.6 where pipx 0.16 has no --global flag; there the global
# locations are set through PIPX_HOME/PIPX_BIN_DIR (see tools/docker/sles.docker).
PIPX_GLOBAL_FLAG="--global"
if [[ "${ID}" == "sles" ]]; then
    PIPX_GLOBAL_FLAG=""
fi

pipx install ${PIPX_GLOBAL_FLAG} https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz --include-deps

echo "Dependencies are installed at $PREFIX"

# Install deps with rbuild
rbuild prepare -d $PREFIX -s develop

if [[ ("${ID}" != "sles") ]]; then
    export CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=ON"
    pip3 install -r $REQ_FILE_DIR/requirements-py.txt
fi
