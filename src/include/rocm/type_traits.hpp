#ifndef ROCM_GUARD_ROCM_TYPE_TRAITS_HPP
#define ROCM_GUARD_ROCM_TYPE_TRAITS_HPP

#include <rocm/config.hpp>
#include <rocm/declval.hpp>
#include <rocm/integral_constant.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class T>
struct type_identity
{
    using type = T;
};

template< class T >
using type_identity_t = typename type_identity<T>::type;

template <bool B, class T = void>
struct enable_if
{
};

template <class T>
struct enable_if<true, T>
{
    using type = T;
};

template <bool B, class T = void>
using enable_if_t = typename enable_if<B, T>::type;

template <bool B, class T, class F>
struct conditional
{
    using type = T;
};

template <class T, class F>
struct conditional<false, T, F>
{
    using type = F;
};

template <bool B, class T, class F>
using conditional_t = typename conditional<B, T, F>::type;

template <class T>
struct remove_cv
{
    using type = T;
};

template <class T>
struct remove_cv<const T> : remove_cv<T>
{
};

template <class T>
struct remove_cv<volatile T> : remove_cv<T>
{
};

template <class T>
struct remove_cv<const volatile T> : remove_cv<T>
{
};

template <class T>
using remove_cv_t = typename remove_cv<T>::type;

template <class T>
struct remove_reference
{
    using type = T;
};
template <class T>
struct remove_reference<T&>
{
    using type = T;
};
template <class T>
struct remove_reference<T&&>
{
    using type = T;
};

template <class T>
using remove_reference_t = typename remove_reference<T>::type;

template <class T>
struct add_pointer : type_identity<remove_reference_t<T>*>
{
};

template <class T>
using add_pointer_t = typename add_pointer<T>::type;

template <class... Ts>
struct common_type;

template <class T>
struct common_type<T>
{
    using type = T;
};

template <class T, class U>
struct common_type<T, U>
{
    using type = decltype(true ? declval<T>() : declval<U>());
};

template <class T, class U, class... Us>
struct common_type<T, U, Us...>
{
    using type = typename common_type<typename common_type<T, U>::type, Us...>::type;
};

template <class... Ts>
using common_type_t = typename common_type<Ts...>::type;

template<class...>
using void_t = void;

// NOLINTNEXTLINE
#define MIGRAPHX_BUILTIN_TYPE_TRAIT1(name)   \
    template <class T>                       \
    struct name : bool_constant<__##name(T)> \
    {                                        \
    }

// NOLINTNEXTLINE
#define MIGRAPHX_BUILTIN_TYPE_TRAIT2(name)      \
    template <class T, class U>                 \
    struct name : bool_constant<__##name(T, U)> \
    {                                           \
    }

// NOLINTNEXTLINE
#define MIGRAPHX_BUILTIN_TYPE_TRAITN(name)       \
    template <class... Ts>                       \
    struct name : bool_constant<__##name(Ts...)> \
    {                                            \
    }

// MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_arithmetic);
// MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_destructible);
// MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_nothrow_destructible);
// MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_pointer);
// MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_scalar);
// MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_signed);
// MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_void);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_abstract);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_aggregate);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_array);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_class);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_compound);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_const);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_empty);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_enum);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_final);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_floating_point);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_function);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_fundamental);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_integral);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_literal_type);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_lvalue_reference);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_member_function_pointer);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_member_object_pointer);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_member_pointer);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_object);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_pod);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_polymorphic);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_reference);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_rvalue_reference);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_standard_layout);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_trivial);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_trivially_copyable);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_trivially_destructible);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_union);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_unsigned);
MIGRAPHX_BUILTIN_TYPE_TRAIT1(is_volatile);
MIGRAPHX_BUILTIN_TYPE_TRAIT2(is_assignable);
MIGRAPHX_BUILTIN_TYPE_TRAIT2(is_base_of);
MIGRAPHX_BUILTIN_TYPE_TRAIT2(is_convertible);
MIGRAPHX_BUILTIN_TYPE_TRAIT2(is_nothrow_assignable);
MIGRAPHX_BUILTIN_TYPE_TRAIT2(is_same);
MIGRAPHX_BUILTIN_TYPE_TRAIT2(is_trivially_assignable);
MIGRAPHX_BUILTIN_TYPE_TRAITN(is_constructible);
MIGRAPHX_BUILTIN_TYPE_TRAITN(is_nothrow_constructible);
MIGRAPHX_BUILTIN_TYPE_TRAITN(is_trivially_constructible);

template <class T>
struct is_void : is_same<void, remove_cv_t<T>>
{
};

#if 0
constexpr unsigned long int_max(unsigned long n)
{
    // Note, left shift cannot be used to get the maximum value of int64_type or
    // uint64_type because it is undefined behavior to left shift 64 bits for
    // these types
    if(n == sizeof(int64_t))
        return -1;
    return (1ul << (n * 8)) - 1;
}

template <class T,
          MIGRAPHX_REQUIRES(is_integral<T>{} or is_floating_point<T>{} or
                            is_same<T, migraphx::half>{})>
constexpr T numeric_max()
{
    if constexpr(is_integral<T>{})
    {
        if constexpr(is_unsigned<T>{})
            return int_max(sizeof(T));
        else
            return int_max(sizeof(T)) / 2;
    }
    else if constexpr(is_same<T, double>{})
        return __DBL_MAX__;
    else if constexpr(is_same<T, float>{})
        return __FLT_MAX__;
    else if constexpr(is_same<T, migraphx::half>{})
        return __FLT16_MAX__;
    else
        return 0;
}

template <class T>
constexpr auto numeric_lowest() -> decltype(numeric_max<T>())
{
    if constexpr(is_integral<T>{})
    {
        if constexpr(is_unsigned<T>{})
            return 0;
        else
            return -numeric_max<T>() - 1;
    }
    else
    {
        return -numeric_max<T>();
    }
}
#endif

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_TYPE_TRAITS_HPP
