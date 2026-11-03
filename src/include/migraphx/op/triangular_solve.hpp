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

#ifndef MIGRAPHX_GUARD_OPERATORS_TRIANGULAR_SOLVE_HPP
#define MIGRAPHX_GUARD_OPERATORS_TRIANGULAR_SOLVE_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/dyn_output.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

// Solves A X = B over the trailing (n, n) and (n, k) matrices of A and B by
// substitution, reading only the lower (or upper) triangle of A; with
// unit_diagonal the diagonal counts as 1 and is not read either
struct triangular_solve
{
    bool lower         = true;
    bool unit_diagonal = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.lower, "lower"), f(self.unit_diagonal, "unit_diagonal"));
    }

    std::string name() const { return "triangular_solve"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2).same_type().same_ndims();
        const auto& a = inputs.front();
        const auto& b = inputs.back();
        if(a.ndim() < 2)
            MIGRAPHX_THROW("TRIANGULAR_SOLVE: inputs must be at least rank 2");
        auto n = a.lens().at(a.ndim() - 2);
        if(a.lens().back() != n)
            MIGRAPHX_THROW("TRIANGULAR_SOLVE: A must be square in its last two dims");
        if(b.lens().at(b.ndim() - 2) != n)
            MIGRAPHX_THROW("TRIANGULAR_SOLVE: B must have as many rows as A");
        if(not std::equal(a.lens().begin(), std::prev(a.lens().end(), 2), b.lens().begin()))
            MIGRAPHX_THROW("TRIANGULAR_SOLVE: batch dims of A and B must match");
        return {b.type(), b.lens()};
    }

    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        argument result{dyn_out.computed_shape};
        visit_all(result, args[0], args[1])([&](auto x, auto a, auto b) {
            const auto& lens = x.get_shape().lens();
            auto ndim        = lens.size();
            auto n           = lens[ndim - 2];
            // one solve per (batch, column), walking the rows in solve order
            auto lane_lens      = lens;
            lane_lens[ndim - 2] = 1;
            shape lanes{x.get_shape().type(), lane_lens};
            par_for(lanes.elements(), [&](auto i) {
                auto xi = lanes.multi(i); // (batch..., 0, col)
                auto ai = xi;             // (batch..., t, s)
                for(std::size_t step = 0; step < n; step++)
                {
                    auto t       = lower ? step : n - 1 - step;
                    xi[ndim - 2] = t;
                    ai[ndim - 2] = t;
                    double acc   = b(xi.begin(), xi.end());
                    for(std::size_t j = 0; j < step; j++)
                    {
                        auto s       = lower ? j : n - 1 - j;
                        ai[ndim - 1] = s;
                        auto xs      = xi;
                        xs[ndim - 2] = s;
                        acc -= a(ai.begin(), ai.end()) * x(xs.begin(), xs.end());
                    }
                    if(not unit_diagonal)
                    {
                        ai[ndim - 1] = t;
                        acc /= a(ai.begin(), ai.end());
                    }
                    x(xi.begin(), xi.end()) = acc;
                }
            });
        });
        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
