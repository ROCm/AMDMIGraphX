#ifndef MIGRAPHX_GUARD_MIGRAPHX_COMPILE_SRC_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_COMPILE_SRC_HPP

#include <migraphx/config.hpp>
#include <migraphx/filesystem.hpp>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct src_file
{
    fs::path path;
    std::pair<const char*, const char*> content;
    std::size_t len() const { return content.second - content.first; }
};

struct src_compiler
{
    std::string compiler                      = "c++";
    std::string flags                         = "";
    std::string output                        = "";
    std::function<fs::path(fs::path)> process = nullptr;
    std::vector<char> compile(const std::vector<src_file>& srcs) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_COMPILE_SRC_HPP
