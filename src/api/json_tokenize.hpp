#ifndef MIGRAPHX_GUARD_API_RTGLIB_JSONIZE_ATTR_STRING_HPP
#define MIGRAPHX_GUARD_API_RTGLIB_JSONIZE_ATTR_STRING_HPP

#include <string>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

using siter = std::string::const_iterator;
using token = std::pair<siter, siter>;

std::string json_tokenize(const std::string& s);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
