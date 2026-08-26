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
#include <migraphx/sym_substitute.hpp>
#include <migraphx/algorithm.hpp>
#include <algorithm>
#include <iterator>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace detail {

expr_map as_expr_map(const symbol_map& symbols)
{
    expr_map result;
    std::transform(
        symbols.begin(), symbols.end(), std::inserter(result, result.end()), [](const auto& p) {
            return std::make_pair(p.first, sym::lit(static_cast<int64_t>(p.second)));
        });
    return result;
}

void substitute(sym::expr& e, const expr_map& symbols)
{
    if(e.empty())
        return;
    e = e.subs(symbols);
}

void substitute(shape::dynamic_dimension& dd, const expr_map& symbols)
{
    if(not dd.is_symbolic())
        return;
    auto e = dd.sym_expr.subs(symbols);
    // A dimension that resolved to a single value is no longer symbolic; carrying it as a fixed
    // range is what lets the enclosing shape collapse to a static one.
    if(auto value = sym::fixed_value(e))
    {
        auto size = sym::to<std::size_t>(*value);
        dd        = shape::dynamic_dimension{size, size};
    }
    else
    {
        dd = shape::dynamic_dimension{e};
    }
}

void substitute(dim_like& d, const expr_map& symbols)
{
    if(not std::holds_alternative<shape::dynamic_dimension>(d))
        return;
    auto dd = std::get<shape::dynamic_dimension>(d);
    substitute(dd, symbols);
    if(dd.is_fixed())
        d = dim_like{static_cast<int64_t>(dd.get_interval().min)};
    else
        d = dim_like{dd};
}

void substitute(shape& s, const expr_map& symbols)
{
    if(not s.sub_shapes().empty())
    {
        std::vector<shape> subs = s.sub_shapes();
        std::for_each(subs.begin(), subs.end(), [&](shape& sub) { substitute(sub, symbols); });
        s = shape{subs};
        return;
    }
    if(not s.dynamic())
        return;

    auto dims = s.dyn_dims();
    std::for_each(
        dims.begin(), dims.end(), [&](shape::dynamic_dimension& dd) { substitute(dd, symbols); });
    auto strides = s.dyn_strides();
    std::for_each(strides.begin(), strides.end(), [&](sym::expr& e) { substitute(e, symbols); });

    auto all_resolved =
        std::all_of(dims.begin(), dims.end(), [](const auto& dd) { return dd.is_fixed(); }) and
        std::all_of(strides.begin(), strides.end(), [](const auto& e) {
            return sym::fixed_value(e).has_value();
        });
    if(not all_resolved)
    {
        s = shape{s.type(), std::move(dims), std::move(strides)};
        return;
    }

    std::vector<std::size_t> lens(dims.size());
    std::transform(dims.begin(), dims.end(), lens.begin(), [](const auto& dd) {
        return dd.get_interval().min;
    });
    if(strides.empty())
    {
        s = shape{s.type(), std::move(lens)};
        return;
    }
    std::vector<std::size_t> static_strides(strides.size());
    std::transform(strides.begin(), strides.end(), static_strides.begin(), [](const auto& e) {
        return sym::to<std::size_t>(*sym::fixed_value(e));
    });
    s = shape{s.type(), std::move(lens), std::move(static_strides)};
}

} // namespace detail

shape substitute_symbols(shape s, const symbol_map& symbols)
{
    detail::substitute(s, detail::as_expr_map(symbols));
    return s;
}

optional<std::vector<std::size_t>> fixed_lens(const std::vector<shape::dynamic_dimension>& dims)
{
    if(dims.empty() or
       not std::all_of(dims.begin(), dims.end(), [](const auto& dd) { return dd.is_fixed(); }))
        return nullopt;
    std::vector<std::size_t> lens(dims.size());
    std::transform(dims.begin(), dims.end(), lens.begin(), [](const auto& dd) {
        return dd.get_interval().min;
    });
    return lens;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
