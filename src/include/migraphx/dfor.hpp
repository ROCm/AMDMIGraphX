#ifndef MIGRAPH_GUARD_MIGRAPHLIB_DFOR_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_DFOR_HPP

#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {

// Multidimensional for loop
inline auto dfor()
{
    return [](auto f) { f(); };
}

template <class T, class... Ts>
auto dfor(T x, Ts... xs)
{
    return [=](auto f) {
        for(T i = 0; i < x; i++)
        {
            dfor(xs...)([&](Ts... is) { f(i, is...); });
        }
    };
}

} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
