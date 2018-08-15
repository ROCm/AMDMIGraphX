#include <migraph/env.hpp>
#include <migraph/ranges.hpp>
#include <cstdlib>

namespace migraph {

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

std::vector<std::string> env(const char* name)
{
    auto p = std::getenv(name);
    if(p == nullptr)
        return {};
    else
        return {{p}};
}

} // namespace migraph
