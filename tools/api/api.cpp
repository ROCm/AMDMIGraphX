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
#include <cstdarg>

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

void set_offload_copy(compile_options& options, bool value) { options.offload_copy = value; }

void set_fast_math(compile_options& options, bool value) { options.fast_math = value; }

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

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
#endif

operation create_op(const char* name, const char* attributes, va_list vlist)
{
    std::string sattributes = attributes == nullptr ? "" : attributes;
    std::vector<char> buffer(sattributes.size() * 2);
    std::vsnprintf(buffer.data(), buffer.size(), sattributes.c_str(), vlist);
    value v = value::object{};
    if(attributes != nullptr)
    {
        v = from_json_string(convert_to_json(std::string(buffer.data())));
    }
    auto op = make_op(name, v);

    return op;
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

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

<% generate_c_api_body() %>
