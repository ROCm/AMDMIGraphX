#ifndef MIGRAPHX_GUARD_RTGLIB_LOAD_SAVE_HPP
#define MIGRAPHX_GUARD_RTGLIB_LOAD_SAVE_HPP

#include <migraphx/program.hpp>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct file_options
{
    std::string format = "msgpack";
};

program load(const std::string& filename, const file_options& options = file_options{});
program load_buffer(const std::vector<char>& buffer, const file_options& options = file_options{});
program
load_buffer(const char* buffer, std::size_t size, const file_options& options = file_options{});

void save(const program& p,
          const std::string& filename,
          const file_options& options = file_options{});
std::vector<char> save_buffer(const program& p, const file_options& options = file_options{});

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
