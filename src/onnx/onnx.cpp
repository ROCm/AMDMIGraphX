#include <migraphx/onnx/onnx_parser.hpp>
#include <migraphx/onnx/op_parser.hpp>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <functional>
#include <array>
#include <utility>
#include <vector>

#include <migraphx/program.hpp>
#include <migraphx/onnx.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class... Ts>
program parse_onnx_from(const onnx_options& options, Ts&&... xs)
{
    onnx::onnx_parser parser;
    parser.map_input_dims         = options.map_input_dims;
    parser.default_dim_value      = options.default_dim_value;
    parser.skip_unknown_operators = options.skip_unknown_operators;
    parser.max_loop_iterations    = options.max_loop_iterations;

    if(options.print_program_on_error)
    {
        // Log the program when it can't be parsed
        try
        {
            parser.parse_from(std::forward<Ts>(xs)...);
        }
        catch(...)
        {
            std::cerr << parser.prog << std::endl;
            throw;
        }
    }
    else
    {
        parser.parse_from(std::forward<Ts>(xs)...);
    }
    return std::move(parser.prog);
}

program parse_onnx(const std::string& name, const onnx_options& options)
{
    std::fstream input(name.c_str(), std::ios::in | std::ios::binary);
    return parse_onnx_from(options, input, name);
}

program parse_onnx_buffer(const std::string& buffer, const onnx_options& options)
{
    return parse_onnx_from(options, buffer.data(), buffer.size());
}

program parse_onnx_buffer(const void* data, std::size_t size, const onnx_options& options)
{
    return parse_onnx_from(options, data, size);
}

std::vector<std::string> get_onnx_operators() { return onnx::get_op_parsers(); }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
