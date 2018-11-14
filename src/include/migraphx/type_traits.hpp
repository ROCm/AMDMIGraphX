/*=============================================================================
    Copyright (c) 2017 Paul Fultz II
    type_traits.hpp
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
==============================================================================*/

#ifndef MIGRAPH_GUARD_RTGLIB_TYPE_TRAITS_HPP
#define MIGRAPH_GUARD_RTGLIB_TYPE_TRAITS_HPP

#include <type_traits>
#include <migraphx/half.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {

#define MIGRAPH_DETAIL_EXTEND_TRAIT_FOR(trait, T) \
    template <class X>                            \
    struct trait : std::trait<X>                  \
    {                                             \
    };                                            \
                                                  \
    template <>                                   \
    struct trait<T> : std::true_type              \
    {                                             \
    };

MIGRAPH_DETAIL_EXTEND_TRAIT_FOR(is_floating_point, half)
MIGRAPH_DETAIL_EXTEND_TRAIT_FOR(is_signed, half)
MIGRAPH_DETAIL_EXTEND_TRAIT_FOR(is_arithmetic, half)

} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
