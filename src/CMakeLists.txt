
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(RegisterOp)

add_library(migraphx 
    auto_contiguous.cpp
    eliminate_common_subexpression.cpp
    decompose.cpp
    propagate_constant.cpp
    dead_code_elimination.cpp
    eliminate_allocation.cpp
    eliminate_contiguous.cpp
    eliminate_concat.cpp
    eliminate_identity.cpp
    eliminate_pad.cpp
    rewrite_batchnorm.cpp
    rewrite_rnn.cpp
    rewrite_pooling.cpp
    env.cpp
    generate.cpp
    instruction.cpp
    load_save.cpp
    make_op.cpp
    msgpack.cpp
    program.cpp
    quantization.cpp
    reduce_dims.cpp
    remap.cpp
    shape.cpp
    schedule.cpp
    serialize.cpp
    pass_manager.cpp
    register_op.cpp
    register_target.cpp
    simplify_algebra.cpp
    simplify_reshapes.cpp
    value.cpp
    verify_args.cpp
    json.cpp
    opt/memory_coloring.cpp
    opt/memory_coloring_impl.cpp
)
rocm_set_soversion(migraphx ${MIGRAPHX_SO_VERSION})
function(register_migraphx_ops)
    foreach(OP ${ARGN})
        register_op(migraphx HEADER migraphx/op/${OP}.hpp OPERATORS op::${OP})
    endforeach()
endfunction()
register_migraphx_ops(
    abs
    acosh
    acos
    add
    argmax
    argmin
    asinh
    asin
    as_shape
    atanh
    atan
    batch_norm_inference
    broadcast
    capture
    ceil
    clip
    concat
    contiguous
    convert
    convolution
    cosh
    cos
    deconvolution
    div
    dot
    elu
    equal
    erf
    exp
    flatten
    floor
    gather
    gru
    identity
    im2col
    leaky_relu
    load
    log
    logsoftmax
    lrn
    lstm
    max
    min
    mul
    multibroadcast
    neg
    outline
    pad
    pooling
    pow
    prelu
    quant_convolution
    quant_dot
    recip
    reduce_max
    reduce_mean
    reduce_min
    reduce_prod
    reduce_sum
    relu
    reshape
    rnn
    rnn_last_cell_output
    rnn_last_hs_output
    rnn_var_sl_last_output
    round
    rsqrt
    scalar
    sigmoid
    sign
    sinh
    sin
    slice
    softmax
    sqdiff
    sqrt
    squeeze
    sub
    tanh
    tan
    transpose
    undefined
    unknown
    unsqueeze
)
register_op(migraphx HEADER migraphx/op/rnn_variable_seq_lens.hpp OPERATORS op::rnn_var_sl_shift_output op::rnn_var_sl_shift_sequence)
register_op(migraphx HEADER migraphx/builtin.hpp OPERATORS builtin::literal builtin::param builtin::returns)
rocm_clang_tidy_check(migraphx)
rocm_install_targets(
  TARGETS migraphx
  INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_path(HALF_INCLUDE_DIR half.hpp)
# TODO: Fix the incorrect path
target_include_directories(migraphx SYSTEM PUBLIC $<BUILD_INTERFACE:${HALF_INCLUDE_DIR}>)

find_package(msgpack REQUIRED)
target_link_libraries(migraphx PRIVATE msgpackc-cxx)
# Make this available to the tests
target_link_libraries(migraphx INTERFACE $<BUILD_INTERFACE:msgpackc-cxx>)

set(PACKAGE_DEPENDS)

add_subdirectory(api)
add_subdirectory(driver)
add_subdirectory(onnx)
add_subdirectory(tf)

add_subdirectory(py)
add_subdirectory(targets/cpu)
if(MIGRAPHX_ENABLE_GPU)
list(APPEND PACKAGE_DEPENDS PACKAGE MIOpen PACKAGE rocblas)
add_subdirectory(targets/gpu)
endif()

rocm_export_targets(
  TARGETS migraphx::migraphx
  NAMESPACE migraphx::
  DEPENDS
    ${PACKAGE_DEPENDS}
)
 

