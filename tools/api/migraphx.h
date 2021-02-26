#ifndef MIGRAPHX_GUARD_C_API_MIGRAPHX_H
#define MIGRAPHX_GUARD_C_API_MIGRAPHX_H

#include <stdlib.h>

// Add new types here
// clang-format off
#define MIGRAPHX_SHAPE_VISIT_TYPES(m) \
    m(bool_type, bool) \
    m(half_type, half) \
    m(float_type, float) \
    m(double_type, double) \
    m(uint8_type, uint8_t) \
    m(int8_type, int8_t) \
    m(uint16_type, uint16_t) \
    m(int16_type, int16_t) \
    m(int32_type, int32_t) \
    m(int64_type, int64_t) \
    m(uint32_type, uint32_t) \
    m(uint64_type, uint64_t)
// clang-format on

#ifdef __cplusplus
extern "C" {
#endif

// return code, more to be added later
typedef enum {
    migraphx_status_success        = 0,
    migraphx_status_bad_param      = 1,
    migraphx_status_unknown_target = 3,
    migraphx_status_unknown_error  = 4,

} migraphx_status;

#define MIGRAPHX_SHAPE_GENERATE_ENUM_TYPES(x, t) migraphx_shape_##x,
/// An enum to represent the different data type inputs
typedef enum {
    MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_SHAPE_GENERATE_ENUM_TYPES)
} migraphx_shape_datatype_t;
#undef MIGRAPHX_SHAPE_GENERATE_ENUM_TYPES

/// Options to be passed when compiling
typedef struct
{
    /// For targets with offloaded memory(such as the gpu), this will insert
    /// instructions during compilation to copy the input parameters to the
    /// offloaded memory and to copy the final result from the offloaded
    /// memory back to main memory.
    bool offload_copy;
    /// Optimize math functions to use faster approximate versions. There may
    /// be slight accuracy degredation when enabled.
    bool fast_math;
} migraphx_compile_options;

/// Options for saving and loading files
typedef struct
{
    /// Format to be used for file. It can either be json or msgpack
    const char* format;
} migraphx_file_options;

<% generate_c_header() %>

#ifdef __cplusplus
}
#endif

#endif
