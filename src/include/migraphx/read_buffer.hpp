#ifndef MIGRAPHX_GUARD_RTGLIB_READ_BUFFER_HPP
#define MIGRAPHX_GUARD_RTGLIB_READ_BUFFER_HPP

#include <migraphx/config.hpp>
#include <migraphx/errors.hpp>
#include <fstream>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::vector<char> read_buffer(const std::string& filename);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
