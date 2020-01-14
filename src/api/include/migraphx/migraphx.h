#ifndef MIGRAPHX_GUARD_C_API_MIGRAPHX_H
#define MIGRAPHX_GUARD_C_API_MIGRAPHX_H

#include <stdlib.h>

/*! Constructs type name from a struct */
#define MIGRAPHX_DECLARE_OBJECT(name) \
    typedef struct                    \
    {                                 \
        void* handle;                 \
    } name;

// Add new types here
// clang-format off
#define MIGRAPHX_SHAPE_VISIT_TYPES(m) \
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
typedef enum {
    MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_SHAPE_GENERATE_ENUM_TYPES)
} migraphx_shape_datatype_t;
#undef MIGRAPHX_SHAPE_GENERATE_ENUM_TYPES

MIGRAPHX_DECLARE_OBJECT(migraphx_shape)

migraphx_status migraphx_shape_create(migraphx_shape* shape,
                                      migraphx_shape_datatype_t type,
                                      const size_t dim_num,
                                      const size_t* dims,
                                      const size_t* strides);

migraphx_status migraphx_shape_destroy(migraphx_shape shape);

migraphx_status migraphx_shape_get(migraphx_shape shape,
                                   migraphx_shape_datatype_t* type,
                                   size_t* dim_num,
                                   const size_t** dims,
                                   const size_t** strides);

MIGRAPHX_DECLARE_OBJECT(migraphx_argument)

migraphx_status
migraphx_argument_create(migraphx_argument* argument, migraphx_shape shape, void* buffer);

migraphx_status migraphx_argument_destroy(migraphx_argument argument);

MIGRAPHX_DECLARE_OBJECT(migraphx_target)

migraphx_status migraphx_target_create(migraphx_target* target, const char* name);

migraphx_status migraphx_target_destroy(migraphx_target target);

migraphx_status
migraphx_target_copy_to(migraphx_target target, migraphx_argument src, migraphx_argument* dst);

migraphx_status
migraphx_target_copy_from(migraphx_target target, migraphx_argument src, migraphx_argument* dst);

MIGRAPHX_DECLARE_OBJECT(migraphx_program)

migraphx_status migraphx_program_create(migraphx_program* program);

migraphx_status migraphx_program_destroy(migraphx_program program);

typedef struct
{
    bool offload_copy;
} migraphx_compile_options;

MIGRAPHX_DECLARE_OBJECT(migraphx_program_parameter_shapes)

migraphx_status migraphx_program_parameter_shapes_create(
    migraphx_program_parameter_shapes* program_parameter_shapes, migraphx_program program);

migraphx_status migraphx_program_parameter_shapes_destroy(
    migraphx_program_parameter_shapes program_parameter_shapes);

migraphx_status
migraphx_program_parameter_shapes_size(migraphx_program_parameter_shapes program_parameter_shapes,
                                       size_t* size);

migraphx_status
migraphx_program_parameter_shapes_names(migraphx_program_parameter_shapes program_parameter_shapes,
                                        const char** names);

migraphx_status migraphx_program_parameter_shapes_get_shape(
    migraphx_program_parameter_shapes program_parameter_shapes,
    const char* name,
    migraphx_shape* shape);

MIGRAPHX_DECLARE_OBJECT(migraphx_program_parameters)

migraphx_status migraphx_program_parameters_create(migraphx_program_parameters* program_parameters);

migraphx_status migraphx_program_parameters_destroy(migraphx_program_parameters program_parameters);

migraphx_status migraphx_program_parameters_add(migraphx_program_parameters program_parameters,
                                                const char* name,
                                                migraphx_argument argument);

migraphx_status migraphx_program_run(migraphx_program program,
                                     migraphx_program_parameters program_parameters,
                                     migraphx_argument* output);

#ifdef __cplusplus
}
#endif

#endif
