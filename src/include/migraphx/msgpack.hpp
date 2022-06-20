#ifndef MIGRAPHX_GUARD_RTGLIB_MSGPACK_HPP
#define MIGRAPHX_GUARD_RTGLIB_MSGPACK_HPP

#include <migraphx/config.hpp>
#include <migraphx/value.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::vector<char> to_msgpack(const value& v);
value from_msgpack(const std::vector<char>& buffer);
value from_msgpack(const char* buffer, std::size_t size);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
