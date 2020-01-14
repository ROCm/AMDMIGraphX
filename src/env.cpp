#include <migraphx/env.hpp>
#include <migraphx/ranges.hpp>
#include <cstdlib>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

bool enabled(const char* name)
{
    auto e = env(name);
    if(e.empty())
        return false;
    return contains({"1", "enable", "enabled", "yes", "true"}, e.front());
}

bool disabled(const char* name)
{
    auto e = env(name);
    if(e.empty())
        return false;
    return contains({"0", "disable", "disabled", "no", "false"}, e.front());
}

std::size_t value_of(const char* name, std::size_t fallback)
{
    auto e = env(name);
    if(e.empty())
        return fallback;
    return std::stoul(e.front());
}

std::vector<std::string> env(const char* name)
{
    auto p = std::getenv(name);
    if(p == nullptr)
        return {};
    else
        return {{p}};
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
