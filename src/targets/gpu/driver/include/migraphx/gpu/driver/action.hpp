#ifndef MIGRAPHX_GUARD_GPU_DRIVER_ACTION_HPP
#define MIGRAPHX_GUARD_GPU_DRIVER_ACTION_HPP

#include <migraphx/config.hpp>
#include <migraphx/auto_register.hpp>
#include <migraphx/type_name.hpp>
#include <migraphx/gpu/driver/parser.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace driver {

using action_function = std::function<void(const parser&, const value&)>;

action_function get_action(const std::string& name);
void register_action(const std::string& name, const action_function& a);

struct auto_register_action
{
    template <class T>
    static void apply()
    {
        auto name = get_type_name<T>();
        register_action(name.substr(name.rfind("::") + 2),
                        [](auto&&... xs) { T::apply(std::forward<decltype(xs)>(xs)...); });
    }
};

template <class T>
using action = auto_register<auto_register_action, T>;

} // namespace driver
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_DRIVER_ACTION_HPP
