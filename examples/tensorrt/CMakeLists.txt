#####################################################################################
# Standalone build for the TensorRT-on-MIGraphX samples.
#
# Unlike the in-tree build, this is its own CMake project: configure it out-of-source
# and point it at a built MIGraphX (the samples link the `commonapi` TensorRT
# adaptation layer plus the MIGraphX libraries it depends on).
#####################################################################################
cmake_minimum_required(VERSION 3.10)
project(tensorrt_samples CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Root of the MIGraphX source repository. examples/tensorrt -> repo root is two up.
get_filename_component(MIGRAPHX_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

# Directory holding the built MIGraphX + commonapi shared libraries. Defaults to the
# in-repo build tree; override with -DMIGRAPHX_LIB_DIR=/path/to/libs when MIGraphX was
# built elsewhere.
set(MIGRAPHX_LIB_DIR "${MIGRAPHX_REPO_ROOT}/build/lib" CACHE PATH
    "Directory containing libmigraphx*/libcommonapi* shared libraries")

list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
find_package(hip REQUIRED)

# The TensorRT adaptation layer (commonapi) and the MIGraphX libraries it needs.
# These must come from the same build as the headers below (matching ABI/version),
# which is why they default to the in-repo build tree rather than an external install.
find_library(COMMONAPI_LIBRARY     NAMES commonapi     HINTS ${MIGRAPHX_LIB_DIR} REQUIRED)
find_library(MIGRAPHX_LIBRARY      NAMES migraphx      HINTS ${MIGRAPHX_LIB_DIR} REQUIRED)
find_library(MIGRAPHX_C_LIBRARY    NAMES migraphx_c    HINTS ${MIGRAPHX_LIB_DIR} REQUIRED)
find_library(MIGRAPHX_ONNX_LIBRARY NAMES migraphx_onnx HINTS ${MIGRAPHX_LIB_DIR} REQUIRED)
find_library(MIGRAPHX_REF_LIBRARY  NAMES migraphx_ref  HINTS ${MIGRAPHX_LIB_DIR} REQUIRED)
find_library(MIGRAPHX_GPU_LIBRARY  NAMES migraphx_gpu  HINTS ${MIGRAPHX_LIB_DIR} REQUIRED)

# Headers used by the samples:
#   - src/common_api/include : public TensorRT API (NvInfer.h, ...)
#   - src/common_api         : internal impl headers (sampleCharRNN reaches into
#                              NetworkDefinition_impl.hpp for debug access)
#   - src/include            : MIGraphX public headers
#   - build/src/include      : MIGraphX build-generated headers (version.h, ...)
set(MIGRAPHX_SAMPLE_INCLUDE_DIRS
    ${MIGRAPHX_REPO_ROOT}/src/include
    ${MIGRAPHX_REPO_ROOT}/build/src/include
    ${MIGRAPHX_REPO_ROOT}/src/common_api/include
    ${MIGRAPHX_REPO_ROOT}/src/common_api
)

set(MIGRAPHX_SAMPLE_LIBS
    ${COMMONAPI_LIBRARY}
    ${MIGRAPHX_ONNX_LIBRARY}
    ${MIGRAPHX_REF_LIBRARY}
    ${MIGRAPHX_GPU_LIBRARY}
    ${MIGRAPHX_C_LIBRARY}
    ${MIGRAPHX_LIBRARY}
    hip::host
)

add_subdirectory(samples)
