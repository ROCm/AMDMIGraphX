#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_ONNX_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_ONNX_HPP

#include <migraphx/program.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/// struct to pass in onnx options to parser
struct onnx_options
{
    /// Old way to set default fixed dimension size (priority over default_dyn_dim_value)
    std::size_t default_dim_value = 0;
    /// Default dynamic dimension size (if not specified in onnx file)
    shape::dynamic_dimension default_dyn_dim_value = {1, 1, 0};
    /// Explicitly specify the dims of an input (priority over map_dyn_input_dims)
    std::unordered_map<std::string, std::vector<std::size_t>> map_input_dims = {};
    /// Explicitly specify dynamic dims of an input
    std::unordered_map<std::string, std::vector<shape::dynamic_dimension>> map_dyn_input_dims = {};
    /// Continue parsing onnx file if an unknown operator is found
    bool skip_unknown_operators = false;
    /// Print program if an error occurs
    bool print_program_on_error = false;
    /// Max iter num for the loop operator
    int64_t max_loop_iterations = 10;
};

/// Create a program from an onnx file
program parse_onnx(const std::string& name, const onnx_options& = onnx_options{});

/// Create a program from an onnx buffer
program parse_onnx_buffer(const std::string& buffer, const onnx_options& options);

/// Create a program from an onnx buffer
program parse_onnx_buffer(const void* data, std::size_t size, const onnx_options& options);

std::vector<std::string> get_onnx_operators();

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
