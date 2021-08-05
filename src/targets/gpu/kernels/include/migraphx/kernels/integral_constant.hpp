#ifndef MIGRAPHX_GUARD_KERNELS_INTEGRAL_CONSTANT_HPP
#define MIGRAPHX_GUARD_KERNELS_INTEGRAL_CONSTANT_HPP

#include <migraphx/kernels/types.hpp>

namespace migraphx {

template <class T, T v>
struct integral_constant
{
    static constexpr T value = v;
    using value_type         = T;
    using type               = integral_constant;
    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }
};

#define MIGRAPHX_INTEGRAL_CONSTANT_BINARY_OP(op)                                \
    template <class T, T v, class U, U w>                                       \
    constexpr inline integral_constant<decltype(v op w), (v op w)> operator op( \
        integral_constant<T, v>, integral_constant<U, w>) noexcept              \
    {                                                                           \
        return {};                                                              \
    }

#define MIGRAPHX_INTEGRAL_CONSTANT_UNARY_OP(op)                             \
    template <class T, T v>                                                 \
    constexpr inline integral_constant<decltype(op v), (op v)> operator op( \
        integral_constant<T, v>) noexcept                                   \
    {                                                                       \
        return {};                                                          \
    }

MIGRAPHX_INTEGRAL_CONSTANT_BINARY_OP(+)
MIGRAPHX_INTEGRAL_CONSTANT_BINARY_OP(-)
MIGRAPHX_INTEGRAL_CONSTANT_BINARY_OP(*)
MIGRAPHX_INTEGRAL_CONSTANT_BINARY_OP(/)
MIGRAPHX_INTEGRAL_CONSTANT_BINARY_OP(%)
MIGRAPHX_INTEGRAL_CONSTANT_BINARY_OP(>>)
MIGRAPHX_INTEGRAL_CONSTANT_BINARY_OP(<<)
MIGRAPHX_INTEGRAL_CONSTANT_BINARY_OP(>)
MIGRAPHX_INTEGRAL_CONSTANT_BINARY_OP(<)
MIGRAPHX_INTEGRAL_CONSTANT_BINARY_OP(<=)
MIGRAPHX_INTEGRAL_CONSTANT_BINARY_OP(>=)
MIGRAPHX_INTEGRAL_CONSTANT_BINARY_OP(==)
MIGRAPHX_INTEGRAL_CONSTANT_BINARY_OP(!=)
MIGRAPHX_INTEGRAL_CONSTANT_BINARY_OP(&)
MIGRAPHX_INTEGRAL_CONSTANT_BINARY_OP (^)
MIGRAPHX_INTEGRAL_CONSTANT_BINARY_OP(|)
MIGRAPHX_INTEGRAL_CONSTANT_BINARY_OP(&&)
MIGRAPHX_INTEGRAL_CONSTANT_BINARY_OP(||)

MIGRAPHX_INTEGRAL_CONSTANT_UNARY_OP(!)
MIGRAPHX_INTEGRAL_CONSTANT_UNARY_OP(~)
MIGRAPHX_INTEGRAL_CONSTANT_UNARY_OP(+)
MIGRAPHX_INTEGRAL_CONSTANT_UNARY_OP(-)

template <bool B>
using bool_constant = integral_constant<bool, B>;

using true_type  = bool_constant<true>;
using false_type = bool_constant<false>;

template <index_int N>
using index_constant = integral_constant<index_int, N>;

template <auto v>
static constexpr auto _c = integral_constant<decltype(v), v>{};

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_INTEGRAL_CONSTANT_HPP
