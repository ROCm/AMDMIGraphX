#ifndef ROCM_GUARD_FUNCTIONAL_OPERATIONS_HPP
#define ROCM_GUARD_FUNCTIONAL_OPERATIONS_HPP

#include <rocm/config.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

#define ROCM_FUNCTIONAL_BINARY_OP(name, op, result) \
    template<class T= void> \
    struct name \
    { \
        constexpr result operator()(const T& x, const T& y) const \
        { \
            return x op y; \
        } \
    }; \
    template<> struct name<void> \
    { \
        using is_transparent = void; \
        template<class T, class U> \
        constexpr auto operator()(T&& x, U&& y) const \
        noexcept(noexcept(static_cast<T&&>(x) op static_cast<U&&>(y))) \
        -> decltype(static_cast<T&&>(x) op static_cast<U&&>(y)) \
        { \
            return static_cast<T&&>(x) op static_cast<U&&>(y); \
        } \
    };

#define ROCM_FUNCTIONAL_UNARY_OP(name, op, result) \
    template<class T= void> \
    struct name \
    { \
        constexpr result operator()(const T& x) const \
        { \
            return op x; \
        } \
    }; \
    template<> struct name<void> \
    { \
        using is_transparent = void; \
        template<class T> \
        constexpr auto operator()(T&& x) const \
        noexcept(noexcept(op static_cast<T&&>(x))) \
        -> decltype(op static_cast<T&&>(x)) \
        { \
            return op static_cast<T&&>(x); \
        } \
    };

ROCM_FUNCTIONAL_BINARY_OP(plus, +, T)
ROCM_FUNCTIONAL_BINARY_OP(minus, -, T)
ROCM_FUNCTIONAL_BINARY_OP(multiplies, *, T)
ROCM_FUNCTIONAL_BINARY_OP(divides, /, T)
ROCM_FUNCTIONAL_BINARY_OP(modulus, %, T)
ROCM_FUNCTIONAL_BINARY_OP(bit_and, &, T)
ROCM_FUNCTIONAL_BINARY_OP(bit_or, |, T)
ROCM_FUNCTIONAL_BINARY_OP(bit_xor, ^, T)

ROCM_FUNCTIONAL_BINARY_OP(equal_to, ==, bool)
ROCM_FUNCTIONAL_BINARY_OP(not_equal_to, !=, bool)
ROCM_FUNCTIONAL_BINARY_OP(greater, >, bool)
ROCM_FUNCTIONAL_BINARY_OP(less, <, bool)
ROCM_FUNCTIONAL_BINARY_OP(greater_equal, >=, bool)
ROCM_FUNCTIONAL_BINARY_OP(less_equal, <=, bool)
ROCM_FUNCTIONAL_BINARY_OP(logical_and, and, bool)
ROCM_FUNCTIONAL_BINARY_OP(logical_or, or, bool)

ROCM_FUNCTIONAL_UNARY_OP(negate, -, T)
ROCM_FUNCTIONAL_UNARY_OP(logical_not, not, bool)
ROCM_FUNCTIONAL_UNARY_OP(bit_not, ~, T)


} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // MIGRAPHX_GUARD_FUNCTIONAL_OPERATIONS_HPP
