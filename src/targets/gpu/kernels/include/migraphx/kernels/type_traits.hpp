#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_KERNELS_TYPE_TRAITS_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_KERNELS_TYPE_TRAITS_HPP

#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/integral_constant.hpp>

namespace migraphx {

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

#define MIGRAPHX_REQUIRES(...) class = enable_if_t<__VA_ARGS__>

} // namespace migraphx

#endif
