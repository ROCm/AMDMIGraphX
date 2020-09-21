#ifndef MIGRAPHX_GUARD_API_RTGLIB_JSONIZE_ATTR_STRING_HPP
#define MIGRAPHX_GUARD_API_RTGLIB_JSONIZE_ATTR_STRING_HPP

#include <string>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::string jsonize_attribute_string(const std::string& op_name, const std::string& str);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif

