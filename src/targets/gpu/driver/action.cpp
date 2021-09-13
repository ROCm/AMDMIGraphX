#include <migraphx/gpu/driver/action.hpp>
#include <migraphx/errors.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace driver {

auto& action_map()
{
    static std::unordered_map<std::string, action_function> m;
    return m;
}

action_function get_action(const std::string& name)
{
    if(action_map().count(name) == 0)
        MIGRAPHX_THROW("Missing action: " + name);
    return action_map().at(name);
}

void register_action(const std::string& name, const action_function& a) { action_map()[name] = a; }

} // namespace driver
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
