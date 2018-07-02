#ifndef MIGRAPH_GUARD_MIGRAPHLIB_DFOR_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_DFOR_HPP

namespace migraph {

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

} // namespace migraph

#endif
