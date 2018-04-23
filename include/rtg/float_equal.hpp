#ifndef GUARD_RTGLIB_FLOAT_EQUAL_HPP
#define RTG_GUARD_RTGLIB_FLOAT_EQUAL_HPP

#include <algorithm>
#include <cmath>
#include <numeric>
#ifdef _MSC_VER
#include <iso646.h>
#endif

namespace rtg {

template <class... Ts>
using common_type = typename std::common_type<Ts...>::type;

struct float_equal_fn
{
    template <class T>
    static bool apply(T x, T y)
    {
        return std::isfinite(x) and std::isfinite(y) and
               std::nextafter(x, std::numeric_limits<T>::lowest()) <= y and
               std::nextafter(x, std::numeric_limits<T>::max()) >= y;
    }

    template <class T, class U>
    bool operator()(T x, U y) const
    {
        return float_equal_fn::apply<common_type<T, U>>(x, y);
    }
};

static constexpr float_equal_fn float_equal{};

} // namespace rtg

#endif
