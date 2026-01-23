#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

set -e

ulimit -c unlimited

# Parse command line arguments
ONNXRT_BRANCH=""
USE_ROCM_FORK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--branch)
            ONNXRT_BRANCH="$2"
            shift 2
            ;;
        --rocm)
            USE_ROCM_FORK=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-b|--branch <branch_name>] [--rocm]"
            exit 1
            ;;
    esac
done

# Save the original directory for file copies later
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Clone onnxruntime if it doesn't exist
if [ ! -d "/onnxruntime" ]; then
    echo "/onnxruntime not found, cloning repository..."
    
    if [ "$USE_ROCM_FORK" = true ]; then
        ONNXRUNTIME_REPO="https://github.com/ROCm/onnxruntime.git"
    else
        ONNXRUNTIME_REPO="https://github.com/Microsoft/onnxruntime"
    fi
    
    # Use specified branch or default to main
    CLONE_BRANCH="${ONNXRT_BRANCH:-main}"
    
    git clone --single-branch --branch "$CLONE_BRANCH" --recursive "$ONNXRUNTIME_REPO" /onnxruntime
    
    cd /onnxruntime
    
    # If no branch was specified and .onnxrt-commit exists, checkout that commit
    if [ -z "$ONNXRT_BRANCH" ] && [ -f "/.onnxrt-commit" ]; then
        echo "Checking out commit from /.onnxrt-commit"
        git checkout "$(cat /.onnxrt-commit)"
    fi
    
    # Run install_common_deps.sh if it exists
    if [ -f "/onnxruntime/dockerfiles/scripts/install_common_deps.sh" ]; then
        echo "Running install_common_deps.sh..."
        /bin/sh /onnxruntime/dockerfiles/scripts/install_common_deps.sh
    fi
    
    # Copy test launchers after clone
    cp "$SCRIPT_DIR/pai_test_launcher.sh" /onnxruntime/tools/ci_build/github/pai/pai_test_launcher.sh 2>/dev/null || :
    [ -f "$SCRIPT_DIR/pai_provider_test_launcher.sh" ] && cp "$SCRIPT_DIR/pai_provider_test_launcher.sh" /onnxruntime/tools/ci_build/github/pai/pai_provider_test_launcher.sh
else
    # /onnxruntime exists - use existing logic
    
    # Copy test launchers before checkout if not switching branches (original behavior)
    if [ -z "$ONNXRT_BRANCH" ]; then
        cp "$SCRIPT_DIR/pai_test_launcher.sh" /onnxruntime/tools/ci_build/github/pai/pai_test_launcher.sh 2>/dev/null || :
        [ -f "$SCRIPT_DIR/pai_provider_test_launcher.sh" ] && cp "$SCRIPT_DIR/pai_provider_test_launcher.sh" /onnxruntime/tools/ci_build/github/pai/pai_provider_test_launcher.sh
    fi

    cd /onnxruntime

    # Checkout specific branch if provided
    if [ -n "$ONNXRT_BRANCH" ]; then
        if [ "$USE_ROCM_FORK" = true ]; then
            echo "Adding ROCm remote and checking out branch: $ONNXRT_BRANCH"
            git remote add rocm https://github.com/ROCm/onnxruntime.git 2>/dev/null || true
            git fetch rocm
            git checkout "$ONNXRT_BRANCH" || git checkout -b "$ONNXRT_BRANCH" "rocm/$ONNXRT_BRANCH"
            git pull rocm "$ONNXRT_BRANCH" || true
        else
            echo "Checking out ONNX Runtime branch: $ONNXRT_BRANCH"
            git fetch origin
            git checkout "$ONNXRT_BRANCH" || git checkout -b "$ONNXRT_BRANCH" "origin/$ONNXRT_BRANCH"
            git pull origin "$ONNXRT_BRANCH" || true
        fi
        # Copy test launchers after checkout to avoid conflicts
        cp "$SCRIPT_DIR/pai_test_launcher.sh" /onnxruntime/tools/ci_build/github/pai/pai_test_launcher.sh 2>/dev/null || :
        [ -f "$SCRIPT_DIR/pai_provider_test_launcher.sh" ] && cp "$SCRIPT_DIR/pai_provider_test_launcher.sh" /onnxruntime/tools/ci_build/github/pai/pai_provider_test_launcher.sh
    fi
fi
pip3 install -r requirements-dev.txt
# Add newer cmake to the path
export PATH="/opt/cmake/bin:$PATH"
export CXXFLAGS="-D__HIP_PLATFORM_AMD__=1 -w"
echo "ONNX Runtime log..."
git log -1
./build.sh --config Release  --cmake_extra_defines CMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ --update --build --build_wheel --parallel --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) --skip_tests --rocm_home /opt/rocm --use_migraphx --migraphx_home /opt/rocm --rocm_version=`cat /opt/rocm/.info/version-dev` --allow_running_as_root --enable_pybind

cd build/Linux/Release
#Add test launcher for onnxrt tests

echo 'InferenceSessionTests.CheckRunProfilerWithSessionOptions' >> ../../../tools/ci_build/github/pai/migraphx-excluded-tests.txt
echo 'InferenceSessionTests.CheckRunProfilerWithSessionOptions2' >> ../../../tools/ci_build/github/pai/migraphx-excluded-tests.txt
echo 'InferenceSessionTests.Test3LayerNestedSubgraph' >> ../../../tools/ci_build/github/pai/migraphx-excluded-tests.txt
echo 'InferenceSessionTests.Test2LayerNestedSubgraph' >> ../../../tools/ci_build/github/pai/migraphx-excluded-tests.txt
../../../tools/ci_build/github/pai/pai_test_launcher.sh || (gdb ./onnxruntime_test_all core -batch -ex bt && exit 1)
../../../tools/ci_build/github/pai/pai_provider_test_launcher.sh || (gdb ./onnxruntime_provider_test core -batch -ex bt && exit 1)
