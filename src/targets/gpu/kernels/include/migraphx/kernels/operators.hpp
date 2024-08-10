#ifndef MIGRAPHX_GUARD_KERNELS_OPERATORS_HPP
#define MIGRAPHX_GUARD_KERNELS_OPERATORS_HPP

#include <migraphx/kernels/functional.hpp>

namespace migraphx {

template<class T>
struct equality_comparable
{
    template<class U>
    friend constexpr auto operator!=(const T& x, const U& y) MIGRAPHX_RETURNS(not (x == y));
    template<class U, class V, MIGRAPHX_REQUIRES(not is_same<T, U>{} and is_same<V, T>{})>
    friend constexpr auto operator!=(const U& x, const V& y) MIGRAPHX_RETURNS(not (x == y));
};

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_OPERATORS_HPP
