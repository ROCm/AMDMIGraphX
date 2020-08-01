#ifndef MIGRAPHX_GUARD_RTGLIB_JSON_HPP
#define MIGRAPHX_GUARD_RTGLIB_JSON_HPP

#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::string to_json_string(const value& val);
value from_json_string(const std::string& str);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
