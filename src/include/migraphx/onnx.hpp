#ifndef MIGRAPH_GUARD_MIGRAPHLIB_ONNX_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_ONNX_HPP

#include <migraphx/program.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {

/// Create a program from an onnx file
program parse_onnx(const std::string& name);

} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
