#ifndef MIGRAPHX_GUARD_KERNELS_OPERATORS_HPP
#define MIGRAPHX_GUARD_KERNELS_OPERATORS_HPP

#include <migraphx/kernels/functional.hpp>

namespace migraphx {

template<class T>
struct equality_comparable
{
    // template<class U, MIGRAPHX_REQUIRES(not is_same<T, U>{})>
    // friend constexpr auto operator==(const U& x, const T& y) MIGRAPHX_RETURNS(y == x);
    template<class U>
    friend constexpr auto operator!=(const T& x, const U& y) MIGRAPHX_RETURNS(not (x == y));
    template<class U, MIGRAPHX_REQUIRES(not is_same<T, U>{})>
    friend constexpr auto operator!=(const U& x, const T& y) MIGRAPHX_RETURNS(not (y == x));
};

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_OPERATORS_HPP
