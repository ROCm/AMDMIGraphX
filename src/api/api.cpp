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

template <class T, class U>
T* object_cast(U* x)
{
    return reinterpret_cast<T*>(x);
}
template <class T, class U>
const T* object_cast(const U* x)
{
    return reinterpret_cast<const T*>(x);
}

extern "C" struct migraphx_shape
{
    migraphx::shape object;
};

extern "C" struct migraphx_argument
{
    migraphx::argument object;
};

extern "C" struct migraphx_target
{
    migraphx::target object;
};

extern "C" struct migraphx_program_parameter_shapes
{
    std::unordered_map<std::string, migraphx::shape> object;
};

extern "C" struct migraphx_program_parameters
{
    std::unordered_map<std::string, migraphx::argument> object;
};

extern "C" struct migraphx_program
{
    migraphx::program object;
};

extern "C" migraphx_status migraphx_shape_destroy(migraphx_shape_t shape)
{
    return migraphx::try_([&] { delete shape; });
}

extern "C" migraphx_status migraphx_shape_create(migraphx_shape_t* shape,
                                                 migraphx_shape_datatype_t type,
                                                 size_t* lengths,
                                                 size_t lengths_size)
{
    return migraphx::try_([&] {
        if(lengths == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter lengths: Null pointer");
        *shape = object_cast<migraphx_shape>(
            new migraphx::shape((migraphx::to_shape_type(type)),
                                (std::vector<size_t>(lengths, lengths + lengths_size))));
    });
}

extern "C" migraphx_status
migraphx_shape_lengths(const size_t** out, size_t* out_size, migraphx_shape_t shape)
{
    return migraphx::try_([&] {
        if(out == nullptr or out_size == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter out: Null pointer");
        if(shape == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter shape: Null pointer");
        auto&& api_result = (shape->object).lens();
        *out              = api_result.data();
        *out_size         = api_result.size();
    });
}

extern "C" migraphx_status
migraphx_shape_strides(const size_t** out, size_t* out_size, migraphx_shape_t shape)
{
    return migraphx::try_([&] {
        if(out == nullptr or out_size == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter out: Null pointer");
        if(shape == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter shape: Null pointer");
        auto&& api_result = (shape->object).strides();
        *out              = api_result.data();
        *out_size         = api_result.size();
    });
}

extern "C" migraphx_status migraphx_shape_type(migraphx_shape_datatype_t* out,
                                               migraphx_shape_t shape)
{
    return migraphx::try_([&] {
        if(out == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter out: Null pointer");
        if(shape == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter shape: Null pointer");
        *out = migraphx::to_shape_type((shape->object).type());
    });
}

extern "C" migraphx_status migraphx_argument_destroy(migraphx_argument_t argument)
{
    return migraphx::try_([&] { delete argument; });
}

extern "C" migraphx_status
migraphx_argument_create(migraphx_argument_t* argument, migraphx_shape_t shape, void* buffer)
{
    return migraphx::try_([&] {
        if(shape == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter shape: Null pointer");
        *argument =
            object_cast<migraphx_argument>(new migraphx::argument((shape->object), (buffer)));
    });
}

extern "C" migraphx_status migraphx_argument_shape(const_migraphx_shape_t* out,
                                                   migraphx_argument_t argument)
{
    return migraphx::try_([&] {
        if(argument == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter argument: Null pointer");
        *out = object_cast<migraphx_shape>(&((argument->object).get_shape()));
    });
}

extern "C" migraphx_status migraphx_argument_buffer(char** out, migraphx_argument_t argument)
{
    return migraphx::try_([&] {
        if(argument == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter argument: Null pointer");
        *out = (argument->object).data();
    });
}

extern "C" migraphx_status migraphx_target_destroy(migraphx_target_t target)
{
    return migraphx::try_([&] { delete target; });
}

extern "C" migraphx_status migraphx_target_create(migraphx_target_t* target, const char* name)
{
    return migraphx::try_([&] {
        *target = object_cast<migraphx_target>(new migraphx::target(migraphx::get_target((name))));
    });
}

extern "C" migraphx_status migraphx_program_parameter_shapes_destroy(
    migraphx_program_parameter_shapes_t program_parameter_shapes)
{
    return migraphx::try_([&] { delete program_parameter_shapes; });
}

extern "C" migraphx_status
migraphx_program_parameter_shapes_size(size_t* out,
                                       migraphx_program_parameter_shapes_t program_parameter_shapes)
{
    return migraphx::try_([&] {
        if(program_parameter_shapes == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param,
                           "Bad parameter program_parameter_shapes: Null pointer");
        *out = (program_parameter_shapes->object).size();
    });
}

extern "C" migraphx_status
migraphx_program_parameter_shapes_get(const_migraphx_shape_t* out,
                                      migraphx_program_parameter_shapes_t program_parameter_shapes,
                                      const char* name)
{
    return migraphx::try_([&] {
        if(program_parameter_shapes == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param,
                           "Bad parameter program_parameter_shapes: Null pointer");
        *out = object_cast<migraphx_shape>(&((program_parameter_shapes->object).get((name))));
    });
}

extern "C" migraphx_status migraphx_program_parameter_shapes_names(
    const char** out, migraphx_program_parameter_shapes_t program_parameter_shapes)
{
    return migraphx::try_([&] {
        if(out == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter out: Null pointer");
        if(program_parameter_shapes == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param,
                           "Bad parameter program_parameter_shapes: Null pointer");
        auto&& api_result = migraphx::get_names((program_parameter_shapes->object));
        std::copy(api_result.begin(), api_result.end(), out);
    });
}

extern "C" migraphx_status
migraphx_program_parameters_destroy(migraphx_program_parameters_t program_parameters)
{
    return migraphx::try_([&] { delete program_parameters; });
}

extern "C" migraphx_status
migraphx_program_parameters_create(migraphx_program_parameters_t* program_parameters)
{
    return migraphx::try_([&] {
        *program_parameters = object_cast<migraphx_program_parameters>(
            new std::unordered_map<std::string, migraphx::argument>());
    });
}

extern "C" migraphx_status
migraphx_program_parameters_add(migraphx_program_parameters_t program_parameters,
                                const char* name,
                                migraphx_argument_t argument)
{
    return migraphx::try_([&] {
        if(program_parameters == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param,
                           "Bad parameter program_parameters: Null pointer");
        if(argument == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter argument: Null pointer");
        (program_parameters->object)[(name)] = (argument->object);
    });
}

extern "C" migraphx_status migraphx_program_destroy(migraphx_program_t program)
{
    return migraphx::try_([&] { delete program; });
}

extern "C" migraphx_status migraphx_program_compile(migraphx_program_t program,
                                                    migraphx_target_t target,
                                                    migraphx_compile_options* options)
{
    return migraphx::try_([&] {
        if(program == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter program: Null pointer");
        if(target == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter target: Null pointer");
        (program->object)
            .compile(
                (target->object),
                (options ? migraphx::to_compile_options(*options) : migraphx::compile_options{}));
    });
}

extern "C" migraphx_status migraphx_program_get_parameter_shapes(migraphx_program_t program)
{
    return migraphx::try_([&] {
        if(program == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter program: Null pointer");
        (program->object).get_parameter_shapes();
    });
}

extern "C" migraphx_status migraphx_program_run(migraphx_argument_t* out,
                                                migraphx_program_t program,
                                                migraphx_program_parameters_t params)
{
    return migraphx::try_([&] {
        if(program == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter program: Null pointer");
        if(params == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter params: Null pointer");
        *out = new migraphx_argument({(program->object).run((params->object))});
    });
}
