#include <migraphx/migraphx.h>
#include <migraphx/rank.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/program.hpp>
#include <migraphx/target.hpp>
#include <migraphx/cpu/target.hpp>

#ifdef HAVE_GPU
#include <migraphx/gpu/target.hpp>
#endif

namespace migraphx {

template <class F>
migraphx_status try_(F f, bool output = true)
{
    try
    {
        f();
    }
    catch(const migraphx::exception& ex)
    {
        if(output)
            std::cerr << "MIGraphX Error: " << ex.what() << std::endl;
        if(ex.error > 0)
            return migraphx_status(ex.error);
        else
            return migraphx_status_unknown_error;
    }
    catch(const std::exception& ex)
    {
        if(output)
            std::cerr << "MIGraphX Error: " << ex.what() << std::endl;
        return migraphx_status_unknown_error;
    }
    catch(...)
    {
        return migraphx_status_unknown_error;
    }
    return migraphx_status_success;
}

shape::type_t to_shape_type(migraphx_shape_datatype_t t)
{
    switch(t)
    {
#define MIGRAPHX_SHAPE_CASE_CONVERT(x, y) \
    case migraphx_shape_##x: return shape::x;
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_SHAPE_CASE_CONVERT)
#undef MIGRAPHX_SHAPE_CASE_CONVERT
    }
    MIGRAPHX_THROW(migraphx_status_bad_param, "Unknown type");
}

migraphx_shape_datatype_t to_shape_type(shape::type_t t)
{
    switch(t)
    {
#define MIGRAPHX_SHAPE_CASE_CONVERT(x, y) \
    case shape::x: return migraphx_shape_##x;
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_SHAPE_CASE_CONVERT)
#undef MIGRAPHX_SHAPE_CASE_CONVERT
    }
    MIGRAPHX_THROW(migraphx_status_bad_param, "Unknown type");
}

target get_target(std::string name)
{
    migraphx::target t;
    if(name == "cpu")
        t = migraphx::cpu::target();
#ifdef HAVE_GPU
    else if(name == "gpu")
        t = migraphx::gpu::target();
#endif
    else
        MIGRAPHX_THROW(migraphx_status_unknown_target, "Unknown target: " + name);
    return t;
}

} // namespace migraphx

template <class T>
T* object_cast(void* x)
{
    return reinterpret_cast<T*>(x);
}
template <class T>
const T* object_cast(const void* x)
{
    return reinterpret_cast<const T*>(x);
}

extern "C" migraphx_status migraphx_shape_destroy(migraphx_shape shape)
{
    return migraphx::try_([&] { delete object_cast<migraphx_shape>(shape.handle); });
}

extern "C" migraphx_status migraphx_shape_create(migraphx_shape* shape,
                                                 migraphx_shape_datatype_t type,
                                                 size_t* lengths,
                                                 size_t lengths_size)
{
    return migraphx::try_([&] {
        if(shape == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter shape: Null pointer");
        if(lengths == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter lengths: Null pointer");
        shape->handle = new migraphx::shape((migraphx::to_shape_type(type)),
                                            (std::vector<size_t>(lengths, lengths + lengths_size)));
    });
}

extern "C" migraphx_status
migraphx_shape_lengths(const size_t** out, size_t* out_size, migraphx_shape shape)
{
    return migraphx::try_([&] {
        if(out == nullptr or out_size == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter out: Null pointer");
        if(shape.handle == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter shape: Null pointer");
        auto&& api_result = (*object_cast<migraphx::shape>(shape.handle)).lens();
        *out              = api_result.data();
        *out_size         = api_result.size();
    });
}

extern "C" migraphx_status
migraphx_shape_strides(const size_t** out, size_t* out_size, migraphx_shape shape)
{
    return migraphx::try_([&] {
        if(out == nullptr or out_size == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter out: Null pointer");
        if(shape.handle == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter shape: Null pointer");
        auto&& api_result = (*object_cast<migraphx::shape>(shape.handle)).strides();
        *out              = api_result.data();
        *out_size         = api_result.size();
    });
}

extern "C" migraphx_status migraphx_shape_type(migraphx_shape_datatype_t* out, migraphx_shape shape)
{
    return migraphx::try_([&] {
        if(out == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter out: Null pointer");
        if(shape.handle == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter shape: Null pointer");
        *out = migraphx::to_shape_type((*object_cast<migraphx::shape>(shape.handle)).type());
    });
}

extern "C" migraphx_status migraphx_argument_destroy(migraphx_argument argument)
{
    return migraphx::try_([&] { delete object_cast<migraphx_argument>(argument.handle); });
}

extern "C" migraphx_status
migraphx_argument_create(migraphx_argument* argument, migraphx_shape shape, void* buffer)
{
    return migraphx::try_([&] {
        if(argument == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter argument: Null pointer");
        if(shape.handle == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter shape: Null pointer");
        argument->handle =
            new migraphx::argument((*object_cast<migraphx::shape>(shape.handle)), (buffer));
    });
}

extern "C" migraphx_status migraphx_argument_shape(migraphx_shape* out, migraphx_argument argument)
{
    return migraphx::try_([&] {
        if(out == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter out: Null pointer");
        if(argument.handle == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter argument: Null pointer");
        out->handle = &((*object_cast<migraphx::argument>(argument.handle)).get_shape());
    });
}

extern "C" migraphx_status migraphx_argument_buffer(char** out, migraphx_argument argument)
{
    return migraphx::try_([&] {
        if(argument.handle == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter argument: Null pointer");
        *out = (*object_cast<migraphx::argument>(argument.handle)).data();
    });
}

extern "C" migraphx_status migraphx_target_destroy(migraphx_target target)
{
    return migraphx::try_([&] { delete object_cast<migraphx_target>(target.handle); });
}

extern "C" migraphx_status migraphx_target_create(migraphx_target* target, const char* name)
{
    return migraphx::try_([&] {
        if(target == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter target: Null pointer");
        target->handle = new migraphx::target(migraphx::get_target((name)));
    });
}

extern "C" migraphx_status migraphx_program_parameter_shapes_destroy(
    migraphx_program_parameter_shapes program_parameter_shapes)
{
    return migraphx::try_([&] {
        delete object_cast<migraphx_program_parameter_shapes>(program_parameter_shapes.handle);
    });
}

extern "C" migraphx_status
migraphx_program_parameter_shapes_size(size_t* out,
                                       migraphx_program_parameter_shapes program_parameter_shapes)
{
    return migraphx::try_([&] {
        if(program_parameter_shapes.handle == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param,
                           "Bad parameter program_parameter_shapes: Null pointer");
        *out = (*object_cast<std::unordered_map<std::string, migraphx::shape>>(
                    program_parameter_shapes.handle))
                   .size();
    });
}

extern "C" migraphx_status
migraphx_program_parameter_shapes_get(migraphx_shape* out,
                                      migraphx_program_parameter_shapes program_parameter_shapes,
                                      const char* name)
{
    return migraphx::try_([&] {
        if(out == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter out: Null pointer");
        if(program_parameter_shapes.handle == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param,
                           "Bad parameter program_parameter_shapes: Null pointer");
        out->handle = &((*object_cast<std::unordered_map<std::string, migraphx::shape>>(
                             program_parameter_shapes.handle))
                            .get((name)));
    });
}

extern "C" migraphx_status
migraphx_program_parameter_shapes_names(const char** out,
                                        migraphx_program_parameter_shapes program_parameter_shapes)
{
    return migraphx::try_([&] {
        if(out == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter out: Null pointer");
        if(program_parameter_shapes.handle == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param,
                           "Bad parameter program_parameter_shapes: Null pointer");
        auto&& api_result =
            migraphx::get_names((*object_cast<std::unordered_map<std::string, migraphx::shape>>(
                program_parameter_shapes.handle)));
        std::copy(api_result.begin(), api_result.end(), out);
    });
}

extern "C" migraphx_status
migraphx_program_parameters_destroy(migraphx_program_parameters program_parameters)
{
    return migraphx::try_(
        [&] { delete object_cast<migraphx_program_parameters>(program_parameters.handle); });
}

extern "C" migraphx_status
migraphx_program_parameters_create(migraphx_program_parameters* program_parameters)
{
    return migraphx::try_([&] {
        if(program_parameters == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param,
                           "Bad parameter program_parameters: Null pointer");
        program_parameters->handle = new std::unordered_map<std::string, migraphx::argument>();
    });
}

extern "C" migraphx_status migraphx_program_parameters_add(
    migraphx_program_parameters program_parameters, const char* name, migraphx_argument argument)
{
    return migraphx::try_([&] {
        if(program_parameters.handle == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param,
                           "Bad parameter program_parameters: Null pointer");
        if(argument.handle == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter argument: Null pointer");
        (*object_cast<std::unordered_map<std::string, migraphx::argument>>(
            program_parameters.handle))[(name)] =
            (*object_cast<migraphx::argument>(argument.handle));
    });
}

extern "C" migraphx_status migraphx_program_destroy(migraphx_program program)
{
    return migraphx::try_([&] { delete object_cast<migraphx_program>(program.handle); });
}

extern "C" migraphx_status migraphx_program_compile(migraphx_program program,
                                                    migraphx_target target,
                                                    migraphx_compile_options* options)
{
    return migraphx::try_([&] {
        if(program.handle == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter program: Null pointer");
        if(target.handle == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter target: Null pointer");
        (*object_cast<migraphx::program>(program.handle))
            .compile(
                (*object_cast<migraphx::target>(target.handle)),
                (options ? migraphx::to_compile_options(*options) : migraphx::compile_options{}));
    });
}

extern "C" migraphx_status migraphx_program_get_parameter_shapes(migraphx_program program)
{
    return migraphx::try_([&] {
        if(program.handle == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter program: Null pointer");
        (*object_cast<migraphx::program>(program.handle)).get_parameter_shapes();
    });
}

extern "C" migraphx_status migraphx_program_run(migraphx_argument* out,
                                                migraphx_program program,
                                                migraphx_program_parameters params)
{
    return migraphx::try_([&] {
        if(out == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter out: Null pointer");
        if(program.handle == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter program: Null pointer");
        if(params.handle == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter params: Null pointer");
        out->handle = new migraphx::argument(
            (*object_cast<migraphx::program>(program.handle))
                .run((*object_cast<std::unordered_map<std::string, migraphx::argument>>(
                    params.handle))));
    });
}
