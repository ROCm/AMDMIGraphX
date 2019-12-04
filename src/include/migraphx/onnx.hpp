#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_ONNX_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_ONNX_HPP

#include <migraphx/program.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/// Create a program from an onnx file
program parse_onnx(const std::string& name, unsigned int batch_size = 1);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
