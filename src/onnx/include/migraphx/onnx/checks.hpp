#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_ONNX_CHECKS_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_ONNX_CHECKS_HPP

#include <migraphx/config.hpp>
#include <migraphx/argument.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

void check_arg_empty(const argument& arg, const std::string& msg);
void check_attr_sizes(size_t kdims, size_t attr_size, const std::string& error_msg);

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
