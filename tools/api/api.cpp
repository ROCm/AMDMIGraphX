#include <migraphx/migraphx.h>
#include <migraphx/rank.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/program.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/target.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/quantization.hpp>

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
                               std::vector<std::size_t> dims)
{
    options.map_input_dims[std::string(name)] = std::move(dims);
}

template <class Value>
std::vector<const char*> get_names(const std::unordered_map<std::string, Value>& m)
{
    std::vector<const char*> result;
    std::transform(
        m.begin(), m.end(), std::back_inserter(result), [](auto&& p) { return p.first.c_str(); });
    return result;
}

void quantize_fp16_wrap(program& prog, std::vector<std::string>& names)
{
    if(names.empty())
    {
        names = {"all"};
    }

    migraphx::quantize_fp16(prog, names);
}

struct quantize_options
{
    std::vector<program::parameter_map> calibration = {};
    std::vector<std::string> op_names               = {};
};

void add_op_name(quantize_options& options, const char* name) { options.op_names.push_back(name); }

void add_calibration_data(quantize_options& options, program::parameter_map& data)
{
    options.calibration.push_back(data);
}

void quantize_int8(program& prog, const target& t, quantize_options& options)
{
    if(options.op_names.empty())
    {
        options.op_names = {"dot", "convolution"};
    }

    migraphx::quantize_int8(prog, t, options.calibration, options.op_names);
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

<% generate_c_api_body() %>
