#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_KERNELS_TYPE_TRAITS_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_KERNELS_TYPE_TRAITS_HPP

#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/integral_constant.hpp>

namespace migraphx {

template <class T>
struct type_identity
{
    using type = T;
};

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

template <class From, class To>
struct is_convertible : bool_constant<__is_convertible(From, To)>
{
};

template <class T, class U>
struct is_same : false_type
{
};

template <class T>
struct is_same<T, T> : true_type
{
};

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
struct add_pointer : type_identity<typename remove_reference<T>::type*>
{
};

template <class T>
using add_pointer_t = typename add_pointer<T>::type;

#define MIGRAPHX_REQUIRES(...) class = enable_if_t<__VA_ARGS__>

} // namespace migraphx

#endif
