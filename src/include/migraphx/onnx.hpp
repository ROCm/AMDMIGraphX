#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_ONNX_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_ONNX_HPP

#include <migraphx/program.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/// struct to pass in onnx options to parser
struct onnx_options
{
    unsigned int batch_size = 1;
};

/// Create a program from an onnx file
program parse_onnx(const std::string& name, onnx_options options);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
