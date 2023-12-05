#ifndef ROCM_GUARD_ROCM_INTEGRAL_CONSTANT_HPP
#define ROCM_GUARD_ROCM_INTEGRAL_CONSTANT_HPP

#include <rocm/config.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

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
#define ROCM_INTEGRAL_CONSTANT_BINARY_OP(op)                                \
    template <class T, T V, class U, U w>                                       \
    constexpr inline integral_constant<decltype(V op w), (V op w)> operator op( \
        integral_constant<T, V>, integral_constant<U, w>) noexcept              \
    {                                                                           \
        return {};                                                              \
    }

// NOLINTNEXTLINE
#define ROCM_INTEGRAL_CONSTANT_UNARY_OP(op)                             \
    template <class T, T V>                                                 \
    constexpr inline integral_constant<decltype(op V), (op V)> operator op( \
        integral_constant<T, V>) noexcept                                   \
    {                                                                       \
        return {};                                                          \
    }

ROCM_INTEGRAL_CONSTANT_BINARY_OP(+)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(-)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(*)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(/)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(%)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(>>)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(<<)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(>)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(<)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(<=)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(>=)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(==)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(!=)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(&)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(^)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(|)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(and)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(or)

ROCM_INTEGRAL_CONSTANT_UNARY_OP(not )
ROCM_INTEGRAL_CONSTANT_UNARY_OP(~)
ROCM_INTEGRAL_CONSTANT_UNARY_OP(+)
ROCM_INTEGRAL_CONSTANT_UNARY_OP(-)

template <bool B>
using bool_constant = integral_constant<bool, B>;

using true_type  = bool_constant<true>;
using false_type = bool_constant<false>;

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_INTEGRAL_CONSTANT_HPP
