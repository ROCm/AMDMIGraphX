#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_FLOAT_EQUAL_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_FLOAT_EQUAL_HPP

#include <algorithm>
#include <cmath>
#include <numeric>
#ifdef _MSC_VER
#include <iso646.h>
#endif

#include <migraphx/requires.hpp>
#include <migraphx/config.hpp>
#include <migraphx/type_traits.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class... Ts>
using common_type = typename std::common_type<Ts...>::type;

struct float_equal_fn
{
    template <class T, MIGRAPHX_REQUIRES(is_floating_point<T>{})>
    static bool apply(T x, T y)
    {
        return std::isfinite(x) and std::isfinite(y) and
               std::nextafter(x, std::numeric_limits<T>::lowest()) <= y and
               std::nextafter(x, std::numeric_limits<T>::max()) >= y;
    }

    template <class T, MIGRAPHX_REQUIRES(not is_floating_point<T>{})>
    static bool apply(T x, T y)
    {
        return x == y;
    }

    template <class T, class U>
    bool operator()(T x, U y) const
    {
        return float_equal_fn::apply<common_type<T, U>>(x, y);
    }
};

static constexpr float_equal_fn float_equal{};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
