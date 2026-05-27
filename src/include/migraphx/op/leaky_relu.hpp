/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_OPERATORS_LEAKY_RELU_HPP
#define MIGRAPHX_GUARD_OPERATORS_LEAKY_RELU_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/op/unary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct leaky_relu : unary<leaky_relu>
{
    float alpha = 0.01;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.alpha, "alpha"));
    }

    // For 0 <= alpha <= 1 (the common case), `max(x, alpha*x)` is equivalent
    // to the where-based form but compiles to a packed v_pk_mul_f16 +
    // v_pk_max_f16 pair on gfx12 fp16, vs the where form which generates a
    // v_cmp + v_cndmask chain (10× more ops on long vectors).
    // The `static_cast<decltype(${0})>(${alpha})` cast keeps alpha in the
    // operand's type so the multiply stays in fp16 instead of getting
    // promoted to fp32 via the double literal.
    std::string point_op() const
    {
        return alpha >= 0.0f and alpha <= 1.0f
                   ? "${function:max}(${0}, static_cast<decltype(${0})>(${alpha}) * ${0})"
                   : "${function:where}(${0} > 0, ${0}, "
                     "static_cast<decltype(${0})>(${alpha}) * ${0})";
    }

    std::string name() const { return "leaky_relu"; }

    auto apply() const
    {
        return [&](auto x) { return x > 0 ? x : x * alpha; };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
