#include <algorithm>
#include <unordered_map>
#include <migraphx/register_target.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::unordered_map<std::string, target>& target_map()
{
    static std::unordered_map<std::string, target> m; // NOLINT
    return m;
}

void register_target(const target& t) { target_map()[t.name()] = t; }

target make_target(const std::string& name)
{
    if(target_map().find(name) == target_map().end())
    {
        MIGRAPHX_THROW("Requested target " + name + " is not enabled");
    }
    return target_map().at(name);
}

std::vector<std::string> get_targets()
{
    std::vector<std::string> result;
    std::transform(target_map().begin(),
                   target_map().end(),
                   std::back_inserter(result),
                   [&](auto&& p) { return p.first; });
    std::sort(result.begin(), result.end());
    return result;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
