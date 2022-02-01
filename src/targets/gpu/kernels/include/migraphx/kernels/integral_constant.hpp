#ifndef MIGRAPHX_GUARD_KERNELS_INTEGRAL_CONSTANT_HPP
#define MIGRAPHX_GUARD_KERNELS_INTEGRAL_CONSTANT_HPP

#include <migraphx/kernels/types.hpp>

namespace migraphx {

template <class T, T V>
struct integral_constant
{
    static constexpr T value = V;
    using value_type         = T;
    using type               = integral_constant;
    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }
    static constexpr type to() { return {}; }
};

// NOLINTNEXTLINE
#define MIGRAPHX_INTEGRAL_CONSTANT_BINARY_OP(op)                                \
    template <class T, T V, class U, U w>                                       \
    constexpr inline integral_constant<decltype(V op w), (V op w)> operator op( \
        integral_constant<T, V>, integral_constant<U, w>) noexcept              \
    {                                                                           \
        return {};                                                              \
    }

// NOLINTNEXTLINE
#define MIGRAPHX_INTEGRAL_CONSTANT_UNARY_OP(op)                             \
    template <class T, T V>                                                 \
    constexpr inline integral_constant<decltype(op V), (op V)> operator op( \
        integral_constant<T, V>) noexcept                                   \
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

template <auto V>
static constexpr auto _c = integral_constant<decltype(V), V>{}; // NOLINT

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_INTEGRAL_CONSTANT_HPP
