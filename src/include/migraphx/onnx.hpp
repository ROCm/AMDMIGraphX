#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_ONNX_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_ONNX_HPP

#include <migraphx/program.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/// struct to pass in onnx options to parser
struct onnx_options
{
    std::size_t default_dim_value                                            = 1;
    std::unordered_map<std::string, std::vector<std::size_t>> map_input_dims = {};
    bool skip_unknown_operators = false;
    bool print_program_on_error = false;
};

/// Create a program from an onnx file
program parse_onnx(const std::string& name, const onnx_options& = onnx_options{});

/// Create a program from an onnx buffer
program parse_onnx_buffer(const std::string& buffer, const onnx_options& options);

/// Create a program from an onnx buffer
program parse_onnx_buffer(const void* data, std::size_t size, const onnx_options& options);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
