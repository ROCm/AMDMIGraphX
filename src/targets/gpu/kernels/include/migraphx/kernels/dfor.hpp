#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_KERNELS_DFOR_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_KERNELS_DFOR_HPP

namespace migraphx {

// Multidimensional for loop
inline constexpr auto dfor()
{
    return [](auto f) { f(); };
}

template <class T, class... Ts>
constexpr auto dfor(T x, Ts... xs)
{
    return [=](auto f) {
        for(T i = 0; i < x; i++)
        {
            dfor(xs...)([&](Ts... is) { f(i, is...); });
        }
    };
}

} // namespace migraphx

#endif
