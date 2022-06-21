#ifndef MIGRAPHX_GUARD_MIGRAPHX_SQLITE_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_SQLITE_HPP

#include <migraphx/config.hpp>
#include <migraphx/filesystem.hpp>
#include <memory>
#include <unordered_map>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct sqlite_impl;

struct sqlite
{
    sqlite() = default;
    static sqlite read(const fs::path& p);
    static sqlite write(const fs::path& p);
    std::vector<std::unordered_map<std::string, std::string>> execute(const std::string& s);

    private:
    std::shared_ptr<sqlite_impl> impl;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_SQLITE_HPP
