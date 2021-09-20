#include <migraphx/operation.hpp>
#include <migraphx/migraphx.h>
#include <migraphx/rank.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/program.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/tf.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/ref/target.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/json.hpp>
#include <migraphx/convert_to_json.hpp>
#include <algorithm>

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
    case migraphx_shape_tuple_type: return shape::tuple_type;
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
    case shape::tuple_type: return migraphx_shape_tuple_type;
#define MIGRAPHX_DETAIL_SHAPE_CASE_CONVERT(x, y) \
    case shape::x: return migraphx_shape_##x;
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_DETAIL_SHAPE_CASE_CONVERT)
#undef MIGRAPHX_DETAIL_SHAPE_CASE_CONVERT
    }
    MIGRAPHX_THROW(migraphx_status_bad_param, "Unknown type");
}

target get_target(const std::string& name) { return make_target(name); }

migraphx::compile_options to_compile_options(const migraphx_compile_options& options)
{
    migraphx::compile_options result{};
    result.offload_copy = options.offload_copy;
    result.fast_math    = options.fast_math;
    return result;
}

void set_file_format(file_options& options, const char* format) { options.format = format; }

void set_default_dim_value(onnx_options& options, size_t value)
{
    options.default_dim_value = value;
}

void set_default_loop_iterations(onnx_options& options, int64_t value)
{
    options.max_loop_iterations = value;
}

void set_nhwc(tf_options& options, bool is_nhwc) { options.is_nhwc = is_nhwc; }

void set_default_dim_value(tf_options& options, size_t value) { options.batch_size = value; }

void set_input_parameter_shape(onnx_options& options,
                               const char* name,
                               std::vector<std::size_t> dims)
{
    options.map_input_dims[std::string(name)] = std::move(dims);
}

void set_input_parameter_shape(tf_options& options, const char* name, std::vector<std::size_t> dims)
{
    options.map_input_dims[std::string(name)] = std::move(dims);
}

void set_output_names(tf_options& options, std::vector<const char*> names)
{
    options.output_node_names = std::vector<std::string>(names.begin(), names.end());
}

template <class Value>
std::vector<const char*> get_names(const std::unordered_map<std::string, Value>& m)
{
    std::vector<const char*> result;
    std::transform(
        m.begin(), m.end(), std::back_inserter(result), [](auto&& p) { return p.first.c_str(); });
    return result;
}

void quantize_fp16_with_op_names(program& prog, std::vector<std::string>& names)
{
    if(names.empty())
    {
        names = {"all"};
    }

    migraphx::quantize_fp16(prog, names);
}

struct quantize_int8_options
{
    std::vector<parameter_map> calibration = {};
    std::vector<std::string> op_names      = {};
};

void add_op_name(quantize_int8_options& options, const char* name)
{
    options.op_names.push_back(name);
}

void add_calibration_data(quantize_int8_options& options, parameter_map& data)
{
    options.calibration.push_back(data);
}

void quantize_int8_wrap(program& prog, const target& t, quantize_int8_options& options)
{
    if(options.op_names.empty())
    {
        options.op_names = {"dot", "convolution"};
    }

    migraphx::quantize_int8(prog, t, options.calibration, options.op_names);
}

operation create_op(const char* name, const char* attributes)
{
    value v = value::object{};
    if(attributes != nullptr)
    {
        v = from_json_string(convert_to_json(std::string(attributes)));
    }
    auto op = make_op(name, v);

    return op;
}

template <class T>
bool equal(const T& x, const T& y)
{
    return x == y;
}

std::vector<argument> run(program& p, const parameter_map& params) { return p.eval(params); }

std::vector<shape> get_output_shapes(program& p) { return p.get_output_shapes(); }

void print_program(const program& p) { std::cout << p << std::endl; }

void print_module(const module& m) { std::cout << m << std::endl; }

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

extern "C" struct migraphx_module;
struct migraphx_module
{
    template <class... Ts>
    migraphx_module(Ts&&... xs) : object(std::forward<Ts>(xs)...)
    {
    }
    migraphx::module object;
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

extern "C" struct migraphx_operation;
struct migraphx_operation
{
    template <class... Ts>
    migraphx_operation(Ts&&... xs) : object(std::forward<Ts>(xs)...)
    {
    }
    migraphx::operation object;
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

extern "C" struct migraphx_file_options;
struct migraphx_file_options
{
    template <class... Ts>
    migraphx_file_options(Ts&&... xs) : object(std::forward<Ts>(xs)...)
    {
    }
    migraphx::file_options object;
};

extern "C" struct migraphx_tf_options;
struct migraphx_tf_options
{
    template <class... Ts>
    migraphx_tf_options(Ts&&... xs) : object(std::forward<Ts>(xs)...)
    {
    }
    migraphx::tf_options object;
};

extern "C" struct migraphx_quantize_op_names;
struct migraphx_quantize_op_names
{
    template <class... Ts>
    migraphx_quantize_op_names(Ts&&... xs) : object(std::forward<Ts>(xs)...)
    {
    }
    std::vector<std::string> object;
};

extern "C" struct migraphx_quantize_int8_options;
struct migraphx_quantize_int8_options
{
    template <class... Ts>
    migraphx_quantize_int8_options(Ts&&... xs) : object(std::forward<Ts>(xs)...)
    {
    }
    migraphx::quantize_int8_options object;
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
        if(lengths == nullptr and lengths_size != 0)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter lengths: Null pointer");
        *shape = object_cast<migraphx_shape_t>(
            allocate<migraphx::shape>((migraphx::to_shape_type(type)),
                                      (std::vector<size_t>(lengths, lengths + lengths_size))));
    });
}

extern "C" migraphx_status migraphx_shape_create_with_strides(migraphx_shape_t* shape,
                                                              migraphx_shape_datatype_t type,
                                                              size_t* lengths,
                                                              size_t lengths_size,
                                                              size_t* strides,
                                                              size_t strides_size)
{
    return migraphx::try_([&] {
        if(lengths == nullptr and lengths_size != 0)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter lengths: Null pointer");
        if(strides == nullptr and strides_size != 0)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter strides: Null pointer");
        *shape = object_cast<migraphx_shape_t>(
            allocate<migraphx::shape>((migraphx::to_shape_type(type)),
                                      (std::vector<size_t>(lengths, lengths + lengths_size)),
                                      (std::vector<size_t>(strides, strides + strides_size))));
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

extern "C" migraphx_status migraphx_module_print(const_migraphx_module_t module)
{
    return migraphx::try_([&] {
        if(module == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter module: Null pointer");
        migraphx::print_module((module->object));
    });
}

extern "C" migraphx_status migraphx_program_destroy(migraphx_program_t program)
{
    return migraphx::try_([&] { destroy((program)); });
}

extern "C" migraphx_status migraphx_program_get_main_module(migraphx_module_t* out,
                                                            migraphx_program_t program)
{
    return migraphx::try_([&] {
        if(program == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter program: Null pointer");
        *out = object_cast<migraphx_module_t>((program->object).get_main_module());
    });
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
        migraphx::print_program((program->object));
    });
}

extern "C" migraphx_status migraphx_program_sort(migraphx_program_t program)
{
    return migraphx::try_([&] {
        if(program == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter program: Null pointer");
        (program->object).sort();
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

extern "C" migraphx_status migraphx_operation_destroy(migraphx_operation_t operation)
{
    return migraphx::try_([&] { destroy((operation)); });
}

extern "C" migraphx_status
migraphx_operation_create(migraphx_operation_t* operation, const char* name, const char* attributes)
{
    return migraphx::try_([&] {
        *operation = object_cast<migraphx_operation_t>(
            allocate<migraphx::operation>(migraphx::create_op((name), (attributes))));
    });
}

extern "C" migraphx_status
migraphx_operation_name(char* out, size_t out_size, migraphx_operation_t operation)
{
    return migraphx::try_([&] {
        if(out == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter out: Null pointer");
        if(operation == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter operation: Null pointer");
        auto&& api_result = (operation->object).name();
        auto* it = std::copy_n(api_result.begin(), std::min(api_result.size(), out_size - 1), out);
        *it      = '\0';
    });
}

extern "C" migraphx_status
migraphx_load(migraphx_program_t* out, const char* name, migraphx_file_options_t options)
{
    return migraphx::try_([&] {
        if(options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter options: Null pointer");
        *out = allocate<migraphx_program_t>(migraphx::load((name), (options->object)));
    });
}

extern "C" migraphx_status
migraphx_save(migraphx_program_t p, const char* name, migraphx_file_options_t options)
{
    return migraphx::try_([&] {
        if(p == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter p: Null pointer");
        if(options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter options: Null pointer");
        migraphx::save((p->object), (name), (options->object));
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

extern "C" migraphx_status migraphx_onnx_options_set_input_parameter_shape(
    migraphx_onnx_options_t onnx_options, const char* name, size_t* dims, size_t dims_size)
{
    return migraphx::try_([&] {
        if(onnx_options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter onnx_options: Null pointer");
        if(dims == nullptr and dims_size != 0)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter dims: Null pointer");
        migraphx::set_input_parameter_shape(
            (onnx_options->object), (name), (std::vector<size_t>(dims, dims + dims_size)));
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

extern "C" migraphx_status migraphx_file_options_destroy(migraphx_file_options_t file_options)
{
    return migraphx::try_([&] { destroy((file_options)); });
}

extern "C" migraphx_status migraphx_file_options_create(migraphx_file_options_t* file_options)
{
    return migraphx::try_([&] {
        *file_options = object_cast<migraphx_file_options_t>(allocate<migraphx::file_options>());
    });
}

extern "C" migraphx_status
migraphx_file_options_set_file_format(migraphx_file_options_t file_options, const char* format)
{
    return migraphx::try_([&] {
        if(file_options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter file_options: Null pointer");
        migraphx::set_file_format((file_options->object), (format));
    });
}

extern "C" migraphx_status
migraphx_onnx_options_set_default_loop_iterations(migraphx_onnx_options_t onnx_options,
                                                  int64_t value)
{
    return migraphx::try_([&] {
        if(onnx_options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter onnx_options: Null pointer");
        migraphx::set_default_loop_iterations((onnx_options->object), (value));
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

extern "C" migraphx_status migraphx_tf_options_destroy(migraphx_tf_options_t tf_options)
{
    return migraphx::try_([&] { destroy((tf_options)); });
}

extern "C" migraphx_status migraphx_tf_options_create(migraphx_tf_options_t* tf_options)
{
    return migraphx::try_([&] {
        *tf_options = object_cast<migraphx_tf_options_t>(allocate<migraphx::tf_options>());
    });
}

extern "C" migraphx_status migraphx_tf_options_set_nhwc(migraphx_tf_options_t tf_options,
                                                        bool is_nhwc)
{
    return migraphx::try_([&] {
        if(tf_options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter tf_options: Null pointer");
        migraphx::set_nhwc((tf_options->object), (is_nhwc));
    });
}

extern "C" migraphx_status migraphx_tf_options_set_input_parameter_shape(
    migraphx_tf_options_t tf_options, const char* name, size_t* dims, size_t dims_size)
{
    return migraphx::try_([&] {
        if(tf_options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter tf_options: Null pointer");
        if(dims == nullptr and dims_size != 0)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter dims: Null pointer");
        migraphx::set_input_parameter_shape(
            (tf_options->object), (name), (std::vector<size_t>(dims, dims + dims_size)));
    });
}

extern "C" migraphx_status
migraphx_tf_options_set_default_dim_value(migraphx_tf_options_t tf_options, size_t value)
{
    return migraphx::try_([&] {
        if(tf_options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter tf_options: Null pointer");
        migraphx::set_default_dim_value((tf_options->object), (value));
    });
}

extern "C" migraphx_status migraphx_tf_options_set_output_names(migraphx_tf_options_t tf_options,
                                                                const char** names,
                                                                size_t names_size)
{
    return migraphx::try_([&] {
        if(tf_options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter tf_options: Null pointer");
        if(names == nullptr and names_size != 0)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter names: Null pointer");
        migraphx::set_output_names((tf_options->object),
                                   (std::vector<const char*>(names, names + names_size)));
    });
}

extern "C" migraphx_status
migraphx_parse_tf(migraphx_program_t* out, const char* name, migraphx_tf_options_t options)
{
    return migraphx::try_([&] {
        if(options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter options: Null pointer");
        *out = allocate<migraphx_program_t>(migraphx::parse_tf((name), (options->object)));
    });
}

extern "C" migraphx_status
migraphx_quantize_op_names_destroy(migraphx_quantize_op_names_t quantize_op_names)
{
    return migraphx::try_([&] { destroy((quantize_op_names)); });
}

extern "C" migraphx_status
migraphx_quantize_op_names_create(migraphx_quantize_op_names_t* quantize_op_names)
{
    return migraphx::try_([&] {
        *quantize_op_names =
            object_cast<migraphx_quantize_op_names_t>(allocate<std::vector<std::string>>());
    });
}

extern "C" migraphx_status
migraphx_quantize_op_names_add(migraphx_quantize_op_names_t quantize_op_names, const char* name)
{
    return migraphx::try_([&] {
        if(quantize_op_names == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param,
                           "Bad parameter quantize_op_names: Null pointer");
        (quantize_op_names->object).push_back((name));
    });
}

extern "C" migraphx_status migraphx_quantize_fp16_with_op_names(migraphx_program_t prog,
                                                                migraphx_quantize_op_names_t name)
{
    return migraphx::try_([&] {
        if(prog == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter prog: Null pointer");
        if(name == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter name: Null pointer");
        migraphx::quantize_fp16_with_op_names((prog->object), (name->object));
    });
}

extern "C" migraphx_status migraphx_quantize_fp16(migraphx_program_t prog)
{
    return migraphx::try_([&] {
        if(prog == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter prog: Null pointer");
        migraphx::quantize_fp16((prog->object));
    });
}

extern "C" migraphx_status
migraphx_quantize_int8_options_destroy(migraphx_quantize_int8_options_t quantize_int8_options)
{
    return migraphx::try_([&] { destroy((quantize_int8_options)); });
}

extern "C" migraphx_status
migraphx_quantize_int8_options_create(migraphx_quantize_int8_options_t* quantize_int8_options)
{
    return migraphx::try_([&] {
        *quantize_int8_options = object_cast<migraphx_quantize_int8_options_t>(
            allocate<migraphx::quantize_int8_options>());
    });
}

extern "C" migraphx_status
migraphx_quantize_int8_options_add_op_name(migraphx_quantize_int8_options_t quantize_int8_options,
                                           const char* name)
{
    return migraphx::try_([&] {
        if(quantize_int8_options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param,
                           "Bad parameter quantize_int8_options: Null pointer");
        migraphx::add_op_name((quantize_int8_options->object), (name));
    });
}

extern "C" migraphx_status migraphx_quantize_int8_options_add_calibration_data(
    migraphx_quantize_int8_options_t quantize_int8_options, migraphx_program_parameters_t data)
{
    return migraphx::try_([&] {
        if(quantize_int8_options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param,
                           "Bad parameter quantize_int8_options: Null pointer");
        if(data == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter data: Null pointer");
        migraphx::add_calibration_data((quantize_int8_options->object), (data->object));
    });
}

extern "C" migraphx_status migraphx_quantize_int8(migraphx_program_t prog,
                                                  migraphx_target_t target,
                                                  migraphx_quantize_int8_options_t options)
{
    return migraphx::try_([&] {
        if(prog == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter prog: Null pointer");
        if(target == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter target: Null pointer");
        if(options == nullptr)
            MIGRAPHX_THROW(migraphx_status_bad_param, "Bad parameter options: Null pointer");
        migraphx::quantize_int8_wrap((prog->object), (target->object), (options->object));
    });
}
