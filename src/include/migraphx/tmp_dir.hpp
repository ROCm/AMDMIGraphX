#ifndef MIGRAPHX_GUARD_RTGLIB_TMP_DIR_HPP
#define MIGRAPHX_GUARD_RTGLIB_TMP_DIR_HPP

#include <migraphx/config.hpp>
#include <migraphx/filesystem.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct tmp_dir
{
    fs::path path;
    tmp_dir(const std::string& prefix = "");

    void execute(const std::string& exe, const std::string& args) const;

    tmp_dir(tmp_dir const&) = delete;
    tmp_dir& operator=(tmp_dir const&) = delete;

    ~tmp_dir();
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
