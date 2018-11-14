/*=============================================================================
    Copyright (c) 2017 Paul Fultz II
    half.hpp
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
==============================================================================*/

#ifndef MIGRAPH_GUARD_RTGLIB_HALF_HPP
#define MIGRAPH_GUARD_RTGLIB_HALF_HPP

#include <half.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {

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

} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
