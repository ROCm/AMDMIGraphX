/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <migraphx/permutation.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/sym.hpp>
#include <map>
#include <functional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

shape reorder_shape(const shape& s, const std::vector<int64_t>& permutation)
{
    if(s.symbolic())
        return {s.type(),
                reorder_dims(s.dyn_dims(), permutation),
                reorder_dims(s.dyn_strides(), permutation)};
    return {s.type(), reorder_dims(s.lens(), permutation), reorder_dims(s.strides(), permutation)};
}

std::vector<int64_t> invert_permutation(const std::vector<int64_t>& permutation)
{
    return sort_permutation(permutation, std::less<>{});
}

std::vector<int64_t> find_permutation(const shape& s)
{
    if(s.dynamic() and not s.symbolic())
        MIGRAPHX_THROW("FIND_PERMUTATION: non-symbolic dynamic shapes not supported");
    std::vector<std::int64_t> result(s.ndim());
    std::iota(result.begin(), result.end(), 0);
    if(s.symbolic())
    {
        const auto& strides = s.dyn_strides();
        const auto& dds     = s.dyn_dims();
        std::stable_sort(result.begin(), result.end(), by(std::greater<>{}, [&](auto x) {
                             return std::make_tuple(strides[x],
                                                    dds[x].sym_expr.value_or(sym::lit(dds[x].min)));
                         }));
    }
    else
    {
        std::stable_sort(result.begin(), result.end(), by(std::greater<>{}, [&](auto x) {
                             return std::make_tuple(s.strides()[x], s.lens()[x]);
                         }));
    }
    return result;
}

std::vector<int64_t> find_permutation(const std::vector<shape>& shapes)
{
    if(shapes.empty())
        return {};
    std::map<std::vector<int64_t>, std::size_t> count;
    for(auto&& s : shapes)
    {
        if(s.broadcasted())
            continue;
        count[find_permutation(s)]++;
    }
    if(count.empty())
    {
        std::vector<int64_t> r(shapes.front().ndim());
        std::iota(r.begin(), r.end(), 0);
        return r;
    }
    auto it = std::max_element(
        count.begin(), count.end(), by(std::less<>{}, [](auto&& p) { return p.second; }));
    assert(it != count.end());
    return it->first;
}

/// Normalize shapes by reordering them by their permutation
std::vector<shape> normalize_permutation(const std::vector<shape>& shapes,
                                         std::vector<int64_t>* permutation)
{
    auto result = shapes;
    auto perm   = find_permutation(shapes);
    std::transform(result.begin(), result.end(), result.begin(), [&](auto s) {
        return reorder_shape(s, perm);
    });
    if(permutation != nullptr)
        *permutation = std::move(perm);
    return result;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
