#ifndef MIGRAPHX_GUARD_RTGLIB_COMPILE_HIP_HPP
#define MIGRAPHX_GUARD_RTGLIB_COMPILE_HIP_HPP

#include <migraphx/config.hpp>
#include <migraphx/filesystem.hpp>
#include <string>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct src_file
{
    fs::path path;
    std::pair<const char*, const char*> content;
    std::size_t len() const { return content.second - content.first; }
};

std::vector<std::vector<char>>
compile_hip_src(const std::vector<src_file>& srcs, std::string params, const std::string& arch);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
