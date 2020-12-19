#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_ONNX_CONV_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_ONNX_CONV_HPP

#include <migraphx/config.hpp>
#include <migraphx/value.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

void recalc_conv_attributes(value& v, size_t kdims);

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
