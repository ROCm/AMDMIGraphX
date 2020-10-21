#ifndef MIGRAPHX_GUARD_RTGLIB_REGISTER_OP_HPP
#define MIGRAPHX_GUARD_RTGLIB_REGISTER_OP_HPP

#include <migraphx/config.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/auto_register.hpp>
#include <cstring>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void register_op(const operation& op);
operation load_op(const std::string& name);
std::vector<std::string> get_operators();

template <class T>
void register_op()
{
    register_op(T{});
}

struct register_op_action
{
    template <class T>
    static void apply()
    {
        register_op<T>();
    }
};

template <class T>
using auto_register_op = auto_register<register_op_action, T>;

#define MIGRAPHX_REGISTER_OP(...) MIGRAPHX_AUTO_REGISTER(register_op_action, __VA_ARGS__)

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
