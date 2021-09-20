
include(CheckCXXCompilerFlag)

add_library(migraphx_cpu
    allocate.cpp
    allocation_model.cpp
    binary.cpp
    concat.cpp
    convolution.cpp
    copy.cpp
    deconvolution.cpp
    dnnl.cpp
    eltwise.cpp
    erf.cpp
    fuse_ops.cpp
    gather.cpp
    gemm.cpp
    layernorm.cpp
    logsoftmax.cpp
    lowering.cpp
    lrn.cpp
    preallocate.cpp
    pooling.cpp
    reduction.cpp
    reorder.cpp
    softmax.cpp
    sub.cpp
    target.cpp
    write_literals.cpp
)
set_target_properties(migraphx_cpu PROPERTIES EXPORT_NAME cpu)
rocm_set_soversion(migraphx_cpu ${MIGRAPHX_SO_VERSION})

set(MIGRAPHX_ENABLE_ZENDNN Off CACHE BOOL "")

find_package(Threads)

if(MIGRAPHX_ENABLE_ZENDNN)
    find_path(ZENDNN_INC_PATH zendnn.hpp)
    find_library(ZENDNN_LIB amdZenDNN)
    find_library(BLIS_LIB blis)
else()
    find_package(dnnl REQUIRED)
endif()

rocm_clang_tidy_check(migraphx_cpu)
if(MIGRAPHX_ENABLE_ZENDNN)
    target_compile_definitions(migraphx_cpu PRIVATE -DMIGRAPHX_ENABLE_ZENDNN)
    target_include_directories(migraphx_cpu PRIVATE ${ZENDNN_INC_PATH})
    message(STATUS "ZENDNN_LIB: ${ZENDNN_LIB}")
    target_link_libraries(migraphx_cpu PRIVATE ${BLIS_LIB})
    target_link_libraries(migraphx_cpu PRIVATE ${ZENDNN_LIB})
else()
    target_link_libraries(migraphx_cpu PRIVATE DNNL::dnnl)
endif()
target_link_libraries(migraphx_cpu PRIVATE migraphx Threads::Threads)

find_package(OpenMP)
target_link_libraries(migraphx_cpu PUBLIC OpenMP::OpenMP_CXX)
# Add library path to rpath to workaround issues with our broken packages
foreach(LIBRARY ${OpenMP_CXX_LIBRARIES})
    if(LIBRARY MATCHES "libomp")
        get_filename_component(LIBRARY_PATH "${LIBRARY}" PATH)
        target_link_libraries(migraphx_cpu PUBLIC -Wl,-rpath=${LIBRARY_PATH} -Wl,-rpath-link=${LIBRARY_PATH})
    endif()
endforeach()

target_link_libraries(migraphx_all_targets INTERFACE migraphx_cpu)

rocm_install_targets(
  TARGETS migraphx_cpu
  INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

