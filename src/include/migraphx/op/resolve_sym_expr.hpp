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
#ifndef MIGRAPHX_GUARD_OPERATORS_RESOLVE_SYM_EXPR_HPP
#define MIGRAPHX_GUARD_OPERATORS_RESOLVE_SYM_EXPR_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <migraphx/sym.hpp>
#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 * Evaluate symbolic dimension expressions at runtime. Dynamic ops (e.g. slice) keep their
 * symbolic dim_like bounds as attributes for compile-time shape inference; resolve_sym_expr turns
 * those expressions into the concrete values fed to the op's runtime-tensor inputs.
 *
 * exprs:   symbolic expressions to evaluate.  symbols: the root variables they reference.
 * Inputs:  one scalar int per symbol, in `symbols` order (symbols[i] = args[i]); each is a single
 *          root-dimension value, e.g. an element of a `dimensions_of` output.
 * Output:  a tuple with one 1-D int64 element per expr, element i = eval(exprs[i]), unclamped.
 */
struct resolve_sym_expr
{
    std::vector<sym::expr> exprs{};
    std::vector<sym::expr> symbols{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.exprs, "exprs"), f(self.symbols, "symbols"));
    }

    std::string name() const { return "resolve_sym_expr"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(symbols.size()).nelements(1);
        return shape{std::vector<shape>(exprs.size(), shape{shape::int64_type, {1}})};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        assert(args.size() == symbols.size());
        std::unordered_map<sym::expr, std::size_t> smap;
        for(std::size_t i = 0; i < symbols.size(); ++i)
            smap[symbols[i]] = args[i].at<std::size_t>();
        const auto& sub_shapes = output_shape.sub_shapes();
        assert(sub_shapes.size() == exprs.size());
        std::vector<argument> results(exprs.size());
        std::transform(exprs.begin(),
                       exprs.end(),
                       sub_shapes.begin(),
                       results.begin(),
                       [&](const sym::expr& e, const shape& s) {
                           argument r{s};
                           r.visit([&](auto out) { out[0] = e.eval_uint(smap); });
                           return r;
                       });
        return argument{results};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
