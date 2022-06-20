#ifndef MIGRAPHX_GUARD_RTGLIB_REGISTER_TARGET_HPP
#define MIGRAPHX_GUARD_RTGLIB_REGISTER_TARGET_HPP

#include <migraphx/config.hpp>
#include <migraphx/target.hpp>
#include <migraphx/auto_register.hpp>
#include <cstring>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void register_target(const target& t);
target make_target(const std::string& name);
std::vector<std::string> get_targets();

template <class T>
void register_target()
{
    register_target(T{});
}

struct register_target_action
{
    template <class T>
    static void apply()
    {
        register_target<T>();
    }
};

template <class T>
using auto_register_target = auto_register<register_target_action, T>;

#define MIGRAPHX_REGISTER_TARGET(...) MIGRAPHX_AUTO_REGISTER(register_target_action, __VA_ARGS__)

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
