#ifndef MIGRAPHX_GUARD_RTGLIB_REGISTER_OP_HPP
#define MIGRAPHX_GUARD_RTGLIB_REGISTER_OP_HPP

#include <migraphx/config.hpp>
#include <migraphx/operation.hpp>
#include <cstring>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void register_op(const operation& op);
operation load_op(const std::string& name);
std::vector<std::string> get_operators();

template <class T>
int register_op()
{
    register_op(T{});
    return 0;
}

template <class T>
struct auto_register_op
{
    static int static_register;
    // This typedef ensures that the static member will be instantiated if
    // the class itself is instantiated
    using static_register_type =
        std::integral_constant<decltype(&static_register), &static_register>;
};

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#endif

template <class T>
int auto_register_op<T>::static_register = register_op<T>(); // NOLINT

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#define MIGRAPHX_REGISTER_OP_NAME_DETAIL(x) migraphx_auto_register_##x
#define MIGRAPHX_REGISTER_OP_NAME(x) MIGRAPHX_REGISTER_OP_NAME_DETAIL(x)
#define MIGRAPHX_REGISTER_OP(...)                                                           \
    void MIGRAPHX_REGISTER_OP_NAME(__LINE__)(migraphx::auto_register_op<__VA_ARGS__> x =    \
                                                 migraphx::auto_register_op<__VA_ARGS__>{}) \
        __attribute__((unused));

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
