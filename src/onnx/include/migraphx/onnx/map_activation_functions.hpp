#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_ONNX_MAP_ACTIVATION_FUNCTIONS_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_ONNX_MAP_ACTIVATION_FUNCTIONS_HPP

#include <migraphx/config.hpp>
#include <migraphx/operation.hpp>
#include <unordered_map>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

const std::unordered_map<std::string, operation>& map_activation_functions();

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
