#include <migraphx/migraphx.h>
#include <migraphx/rank.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/program.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/target.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/cpu/target.hpp>

#ifdef HAVE_GPU
#include <migraphx/gpu/target.hpp>
#endif

namespace migraphx {

template <class F>
migraphx_status try_(F f, bool output = true) // NOLINT
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
#define MIGRAPHX_DETAIL_SHAPE_CASE_CONVERT(x, y) \
    case migraphx_shape_##x: return shape::x;
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_DETAIL_SHAPE_CASE_CONVERT)
#undef MIGRAPHX_DETAIL_SHAPE_CASE_CONVERT
    }
    MIGRAPHX_THROW(migraphx_status_bad_param, "Unknown type");
}

migraphx_shape_datatype_t to_shape_type(shape::type_t t)
{
    switch(t)
    {
#define MIGRAPHX_DETAIL_SHAPE_CASE_CONVERT(x, y) \
    case shape::x: return migraphx_shape_##x;
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_DETAIL_SHAPE_CASE_CONVERT)
#undef MIGRAPHX_DETAIL_SHAPE_CASE_CONVERT
    }
    MIGRAPHX_THROW(migraphx_status_bad_param, "Unknown type");
}

target get_target(const std::string& name)
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

migraphx::compile_options to_compile_options(const migraphx_compile_options& options)
{
    migraphx::compile_options result{};
    result.offload_copy = options.offload_copy;
    return result;
}

void set_default_dim_value(onnx_options& options, size_t value)
{
    options.default_dim_value = value;
}

void set_input_parameter_shape(onnx_options& options,
                               const char* name,
                               const size_t* dims,
                               const size_t dim_num)
{
    options.map_input_dims[std::string(name)] = std::vector<std::size_t>(dims, dims + dim_num);
}

template <class Value>
std::vector<const char*> get_names(const std::unordered_map<std::string, Value>& m)
{
    std::vector<const char*> result;
    std::transform(
        m.begin(), m.end(), std::back_inserter(result), [](auto&& p) { return p.first.c_str(); });
    return result;
}

template <class T>
bool equal(const T& x, const T& y)
{
    return x == y;
}

std::vector<argument> run(program& p, const program::parameter_map& params)
{
    return p.eval(params);
}

std::vector<shape> get_output_shapes(program& p) { return p.get_output_shapes(); }

void print(const program& p) { std::cout << p << std::endl; }

} // namespace migraphx

template <class T, class U, class Target = std::remove_pointer_t<T>>
Target* object_cast(U* x)
{
    return reinterpret_cast<Target*>(x);
}
template <class T, class U, class Target = std::remove_pointer_t<T>>
const Target* object_cast(const U* x)
{
    return reinterpret_cast<const Target*>(x);
}

template <class T, class... Ts, class Target = std::remove_pointer_t<T>>
Target* allocate(Ts&&... xs)
{
    return new Target(std::forward<Ts>(xs)...); // NOLINT
}

template <class T>
void destroy(T* x)
{
    delete x; // NOLINT
}

extern "C" struct migraphx_shape;
struct migraphx_shape
{
    template <class... Ts>
    migraphx_shape(Ts&&... xs) : object(std::forward<Ts>(xs)...)
    {
    }
    migraphx::shape object;
};

extern "C" struct migraphx_argument;
struct migraphx_argument
{
    template <class... Ts>
    migraphx_argument(Ts&&... xs) : object(std::forward<Ts>(xs)...)
    {
    }
    migraphx::argument object;
};

extern "C" struct migraphx_target;
struct migraphx_target
{
    template <class... Ts>
    migraphx_target(Ts&&... xs) : object(std::forward<Ts>(xs)...)
    {
    }
    migraphx::target object;
};

extern "C" struct migraphx_program_parameter_shapes;
struct migraphx_program_parameter_shapes
{
    template <class... Ts>
    migraphx_program_parameter_shapes(Ts&&... xs) : object(std::forward<Ts>(xs)...)
    {
    }
    std::unordered_map<std::string, migraphx::shape> object;
};

extern "C" struct migraphx_program_parameters;
struct migraphx_program_parameters
{
    template <class... Ts>
    migraphx_program_parameters(Ts&&... xs) : object(std::forward<Ts>(xs)...)
    {
    }
    std::unordered_map<std::string, migraphx::argument> object;
};

extern "C" struct migraphx_arguments;
struct migraphx_arguments
{
    template <class... Ts>
    migraphx_arguments(Ts&&... xs) : object(std::forward<Ts>(xs)...)
    {
    }
    std::vector<migraphx::argument> object;
};

extern "C" struct migraphx_shapes;
struct migraphx_shapes
{
    template <class... Ts>
    migraphx_shapes(Ts&&... xs) : object(std::forward<Ts>(xs)...)
    {
    }
    std::vector<migraphx::shape> object;
};

extern "C" struct migraphx_program;
struct migraphx_program
{
    template <class... Ts>
    migraphx_program(Ts&&... xs) : object(std::forward<Ts>(xs)...)
    {
    }
    migraphx::program object;
};

extern "C" struct migraphx_onnx_options;
struct migraphx_onnx_options
{
    template <class... Ts>
    migraphx_onnx_options(Ts&&... xs) : object(std::forward<Ts>(xs)...)
    {
    }
    migraphx::onnx_options object;
};

extern "C" migraphx_status migraphx_shape_destroy(migraphx_shape_t shape)
{
    return migraphx::try_([&] { destroy((shape)); });
}

extern "C" migraphx_status migraphx_shape_create(migraphx_shape_t* shape,
                                                 migraphx_shape_datatype_t type,
                                                 size_t* lengths,
                                                 size_t lengths_size)
{
    return migraphx::try_([&] {
        if(lengths == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter lengths: Null pointer");
        *shape = object_cast<migraphx_shape_t>(
            allocate<migraphx::shape>((migraphx::to_shape_type(type)),
                                      (std::vector<size_t>(lengths, lengths + lengths_size))));
    });
}

extern "C" migraphx_status migraphx_shape_create_scalar(migraphx_shape_t* shape,
                                                        migraphx_shape_datatype_t type)
{
    return migraphx::try_([&] {
        *shape = object_cast<migraphx_shape_t>(
            allocate<migraphx::shape>((migraphx::to_shape_type(type))));
    });
}

extern "C" migraphx_status
migraphx_shape_lengths(const size_t** out, size_t* out_size, const_migraphx_shape_t shape)
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
migraphx_shape_strides(const size_t** out, size_t* out_size, const_migraphx_shape_t shape)
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
                                               const_migraphx_shape_t shape)
{
    return migraphx::try_([&] {
        if(out == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter out: Null pointer");
        if(shape == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter shape: Null pointer");
        *out = migraphx::to_shape_type((shape->object).type());
    });
}

extern "C" migraphx_status migraphx_shape_bytes(size_t* out, const_migraphx_shape_t shape)
{
    return migraphx::try_([&] {
        if(shape == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter shape: Null pointer");
        *out = (shape->object).bytes();
    });
}

extern "C" migraphx_status
migraphx_shape_equal(bool* out, const_migraphx_shape_t shape, const_migraphx_shape_t x)
{
    return migraphx::try_([&] {
        if(shape == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter shape: Null pointer");
        if(x == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter x: Null pointer");
        *out = migraphx::equal((shape->object), (x->object));
    });
}

extern "C" migraphx_status migraphx_argument_destroy(migraphx_argument_t argument)
{
    return migraphx::try_([&] { destroy((argument)); });
}

extern "C" migraphx_status
migraphx_argument_create(migraphx_argument_t* argument, const_migraphx_shape_t shape, void* buffer)
{
    return migraphx::try_([&] {
        if(shape == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter shape: Null pointer");
        *argument = object_cast<migraphx_argument_t>(
            allocate<migraphx::argument>((shape->object), (buffer)));
    });
}

extern "C" migraphx_status migraphx_argument_shape(const_migraphx_shape_t* out,
                                                   const_migraphx_argument_t argument)
{
    return migraphx::try_([&] {
        if(argument == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter argument: Null pointer");
        *out = object_cast<const_migraphx_shape_t>(&((argument->object).get_shape()));
    });
}

extern "C" migraphx_status migraphx_argument_buffer(char** out, const_migraphx_argument_t argument)
{
    return migraphx::try_([&] {
        if(argument == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter argument: Null pointer");
        *out = (argument->object).data();
    });
}

extern "C" migraphx_status
migraphx_argument_equal(bool* out, const_migraphx_argument_t argument, const_migraphx_argument_t x)
{
    return migraphx::try_([&] {
        if(argument == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter argument: Null pointer");
        if(x == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter x: Null pointer");
        *out = migraphx::equal((argument->object), (x->object));
    });
}

extern "C" migraphx_status
migraphx_argument_generate(migraphx_argument_t* out, const_migraphx_shape_t s, size_t seed)
{
    return migraphx::try_([&] {
        if(s == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter s: Null pointer");
        *out = allocate<migraphx_argument_t>(migraphx::generate_argument((s->object), (seed)));
    });
}

extern "C" migraphx_status migraphx_target_destroy(migraphx_target_t target)
{
    return migraphx::try_([&] { destroy((target)); });
}

extern "C" migraphx_status migraphx_target_create(migraphx_target_t* target, const char* name)
{
    return migraphx::try_([&] {
        *target = object_cast<migraphx_target_t>(
            allocate<migraphx::target>(migraphx::get_target((name))));
    });
}

extern "C" migraphx_status migraphx_program_parameter_shapes_destroy(
    migraphx_program_parameter_shapes_t program_parameter_shapes)
{
    return migraphx::try_([&] { destroy((program_parameter_shapes)); });
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
        *out =
            object_cast<const_migraphx_shape_t>(&((program_parameter_shapes->object).at((name))));
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
    return migraphx::try_([&] { destroy((program_parameters)); });
}

extern "C" migraphx_status
migraphx_program_parameters_create(migraphx_program_parameters_t* program_parameters)
{
    return migraphx::try_([&] {
        *program_parameters = object_cast<migraphx_program_parameters_t>(
            allocate<std::unordered_map<std::string, migraphx::argument>>());
    });
}

extern "C" migraphx_status
migraphx_program_parameters_add(migraphx_program_parameters_t program_parameters,
                                const char* name,
                                const_migraphx_argument_t argument)
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

extern "C" migraphx_status migraphx_arguments_destroy(migraphx_arguments_t arguments)
{
    return migraphx::try_([&] { destroy((arguments)); });
}

extern "C" migraphx_status migraphx_arguments_size(size_t* out, migraphx_arguments_t arguments)
{
    return migraphx::try_([&] {
        if(arguments == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter arguments: Null pointer");
        *out = (arguments->object).size();
    });
}

extern "C" migraphx_status
migraphx_arguments_get(const_migraphx_argument_t* out, migraphx_arguments_t arguments, size_t idx)
{
    return migraphx::try_([&] {
        if(arguments == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter arguments: Null pointer");
        *out = object_cast<const_migraphx_argument_t>(&((arguments->object).at((idx))));
    });
}

extern "C" migraphx_status migraphx_shapes_destroy(migraphx_shapes_t shapes)
{
    return migraphx::try_([&] { destroy((shapes)); });
}

extern "C" migraphx_status migraphx_shapes_size(size_t* out, migraphx_shapes_t shapes)
{
    return migraphx::try_([&] {
        if(shapes == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter shapes: Null pointer");
        *out = (shapes->object).size();
    });
}

extern "C" migraphx_status
migraphx_shapes_get(const_migraphx_shape_t* out, migraphx_shapes_t shapes, size_t idx)
{
    return migraphx::try_([&] {
        if(shapes == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter shapes: Null pointer");
        *out = object_cast<const_migraphx_shape_t>(&((shapes->object).at((idx))));
    });
}

extern "C" migraphx_status migraphx_program_destroy(migraphx_program_t program)
{
    return migraphx::try_([&] { destroy((program)); });
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
            .compile((target->object),
                     (options == nullptr ? migraphx::compile_options{}
                                         : migraphx::to_compile_options(*options)));
    });
}

extern "C" migraphx_status
migraphx_program_get_parameter_shapes(migraphx_program_parameter_shapes_t* out,
                                      migraphx_program_t program)
{
    return migraphx::try_([&] {
        if(program == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter program: Null pointer");
        *out =
            allocate<migraphx_program_parameter_shapes_t>((program->object).get_parameter_shapes());
    });
}

extern "C" migraphx_status migraphx_program_get_output_shapes(migraphx_shapes_t* out,
                                                              migraphx_program_t program)
{
    return migraphx::try_([&] {
        if(program == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter program: Null pointer");
        *out = allocate<migraphx_shapes_t>(migraphx::get_output_shapes((program->object)));
    });
}

extern "C" migraphx_status migraphx_program_print(const_migraphx_program_t program)
{
    return migraphx::try_([&] {
        if(program == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter program: Null pointer");
        migraphx::print((program->object));
    });
}

extern "C" migraphx_status migraphx_program_run(migraphx_arguments_t* out,
                                                migraphx_program_t program,
                                                migraphx_program_parameters_t params)
{
    return migraphx::try_([&] {
        if(program == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter program: Null pointer");
        if(params == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter params: Null pointer");
        *out = allocate<migraphx_arguments_t>(migraphx::run((program->object), (params->object)));
    });
}

extern "C" migraphx_status
migraphx_program_equal(bool* out, const_migraphx_program_t program, const_migraphx_program_t x)
{
    return migraphx::try_([&] {
        if(program == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter program: Null pointer");
        if(x == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter x: Null pointer");
        *out = migraphx::equal((program->object), (x->object));
    });
}

extern "C" migraphx_status migraphx_onnx_options_destroy(migraphx_onnx_options_t onnx_options)
{
    return migraphx::try_([&] { destroy((onnx_options)); });
}

extern "C" migraphx_status migraphx_onnx_options_create(migraphx_onnx_options_t* onnx_options)
{
    return migraphx::try_([&] {
        *onnx_options = object_cast<migraphx_onnx_options_t>(allocate<migraphx::onnx_options>());
    });
}

extern "C" migraphx_status
migraphx_onnx_options_set_input_parameter_shape(migraphx_onnx_options_t onnx_options,
                                                const char* name,
                                                const size_t* dims,
                                                const size_t dim_num)
{
    return migraphx::try_([&] {
        if(onnx_options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter onnx_options: Null pointer");
        migraphx::set_input_parameter_shape((onnx_options->object), (name), (dims), (dim_num));
    });
}

extern "C" migraphx_status
migraphx_onnx_options_set_default_dim_value(migraphx_onnx_options_t onnx_options, size_t value)
{
    return migraphx::try_([&] {
        if(onnx_options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter onnx_options: Null pointer");
        migraphx::set_default_dim_value((onnx_options->object), (value));
    });
}

extern "C" migraphx_status
migraphx_parse_onnx(migraphx_program_t* out, const char* name, migraphx_onnx_options_t options)
{
    return migraphx::try_([&] {
        if(options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter options: Null pointer");
        *out = allocate<migraphx_program_t>(migraphx::parse_onnx((name), (options->object)));
    });
}

extern "C" migraphx_status migraphx_parse_onnx_buffer(migraphx_program_t* out,
                                                      const void* data,
                                                      size_t size,
                                                      migraphx_onnx_options_t options)
{
    return migraphx::try_([&] {
        if(options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter options: Null pointer");
        *out = allocate<migraphx_program_t>(
            migraphx::parse_onnx_buffer((data), (size), (options->object)));
    });
}
