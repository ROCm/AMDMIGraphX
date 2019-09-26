/*=============================================================================
    Copyright (c) 2017 Paul Fultz II
    permutation.hpp
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
==============================================================================*/

#ifndef MIGRAPHX_GUARD_RTGLIB_PERMUTATION_HPP
#define MIGRAPHX_GUARD_RTGLIB_PERMUTATION_HPP

#include <migraphx/config.hpp>
#include <migraphx/shape.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class Vector>
inline Vector reorder_dims(const Vector& dims, const std::vector<int64_t>& permutation)
{
    Vector result(dims.size());
    assert(dims.size() == permutation.size());
    for(std::size_t i = 0; i < dims.size(); i++)
    {
        result[i] = dims[permutation[i]];
    }
    return result;
}

inline shape reorder_shape(const shape& s, const std::vector<int64_t>& permutation)
{
    return {s.type(), reorder_dims(s.lens(), permutation), reorder_dims(s.strides(), permutation)};
}

template <class Vector, class Op>
inline std::vector<int64_t> sort_permutation(const Vector& data, Op op)
{
    std::vector<std::int64_t> result(data.size());
    std::iota(result.begin(), result.end(), 0);
    std::sort(result.begin(), result.end(), [&](auto x, auto y) { return op(data[x], data[y]); });
    return result;
}

inline std::vector<int64_t> invert_permutation(const std::vector<int64_t>& permutation)
{
    return sort_permutation(permutation, std::less<>{});
}

inline std::vector<int64_t> find_permutation(const shape& s)
{
    return sort_permutation(s.strides(), std::greater<>{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
