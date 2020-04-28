/*=============================================================================
    Copyright (c) 2017 Paul Fultz II
    half.hpp
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
==============================================================================*/

#ifndef MIGRAPHX_GUARD_RTGLIB_HALF_HPP
#define MIGRAPHX_GUARD_RTGLIB_HALF_HPP

#include <half.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

using half = half_float::half;

namespace detail {
template <class T>
struct deduce
{
    using type = T;
};

template <>
struct deduce<half_float::detail::expr>
{
    using type = half;
};
} // namespace detail

template <class T>
using deduce = typename detail::deduce<T>::type;

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

namespace std {

template<class T>
struct common_type<migraphx::half, T>
: std::common_type<float, T>
{};

template<class T>
struct common_type<T, migraphx::half>
: std::common_type<float, T>
{};

template<>
struct common_type<migraphx::half, migraphx::half>
{
    using type = migraphx::half;
};

} // namespace std

#endif
