#include <migraphx/register_target.hpp>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::unordered_map<std::string, target>& target_map()
{
    static std::unordered_map<std::string, target> m;
    return m;
}

void register_target(const target& t) { target_map()[t.name()] = t; }
target make_target(const std::string& name) { return target_map().at(name); }
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
