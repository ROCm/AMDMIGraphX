#ifndef MIGRAPHX_GUARD_RTGLIB_AUTO_REGISTER_HPP
#define MIGRAPHX_GUARD_RTGLIB_AUTO_REGISTER_HPP

#include <migraphx/config.hpp>
#include <type_traits>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class Action, class T>
int auto_register_action()
{
    Action::template apply<T>();
    return 0;
}

template <class Action, class T>
struct auto_register
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

template <class Action, class T>
int auto_register<Action, T>::static_register = auto_register_action<Action, T>(); // NOLINT

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#define MIGRAPHX_AUTO_REGISTER_NAME_DETAIL(x) migraphx_auto_register_##x
#define MIGRAPHX_AUTO_REGISTER_NAME(x) MIGRAPHX_AUTO_REGISTER_NAME_DETAIL(x)
// NOLINTNEXTLINE
#define MIGRAPHX_AUTO_REGISTER(...)                                                        \
    void MIGRAPHX_AUTO_REGISTER_NAME(__LINE__)(migraphx::auto_register<__VA_ARGS__> x =    \
                                                   migraphx::auto_register<__VA_ARGS__>{}) \
        __attribute__((unused));

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
