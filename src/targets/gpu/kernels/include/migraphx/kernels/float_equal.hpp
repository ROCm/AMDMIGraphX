#ifndef MIGRAPHX_GUARD_KERNELS_FLOAT_EQUAL_HPP
#define MIGRAPHX_GUARD_KERNELS_FLOAT_EQUAL_HPP

#include <migraphx/kernels/type_traits.hpp>

namespace migraphx {

template <class T, class U>
constexpr bool float_equal(T x, U y)
{
    if constexpr(is_integral<T>{} or is_integral<U>{})
        return x == y;
    return not(x < y or x > y);
}


} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_FLOAT_EQUAL_HPP
