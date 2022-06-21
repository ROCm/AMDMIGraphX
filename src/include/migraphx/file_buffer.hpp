#ifndef MIGRAPHX_GUARD_RTGLIB_FILE_BUFFER_HPP
#define MIGRAPHX_GUARD_RTGLIB_FILE_BUFFER_HPP

#include <migraphx/config.hpp>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::vector<char> read_buffer(const std::string& filename);
std::string read_string(const std::string& filename);

void write_buffer(const std::string& filename, const char* buffer, std::size_t size);
void write_buffer(const std::string& filename, const std::vector<char>& buffer);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
