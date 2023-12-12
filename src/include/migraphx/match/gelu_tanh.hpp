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
#ifndef MIGRAPHX_GUARD_MATCH_GELU_TANH_HPP
#define MIGRAPHX_GUARD_MATCH_GELU_TANH_HPP

#include <migraphx/config.hpp>
#include <migraphx/matcher.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace match {

namespace detail {
template <class F>
struct gelu_tanh_matcher
{
    F f;

    /// x ^ 3
    auto pow_fn() const { return f("pow")(used_once(), arg(1)(has_value(3.0f))); }

    /// tanh( sqrt(2/M_PI) * (x + 0.044715 * x ^ 3 )
    auto tanh_fn() const
    {
        auto mul_const_pow       = f("mul")(either_arg(0, 1)(has_value(0.044715f), pow_fn()));
        auto add_any_mul         = f("add")(any_arg(0, 1)(mul_const_pow));
        auto either_SQRT2RPI_add = either_arg(0, 1)(has_value(sqrt(M_2_PI)), add_any_mul);
        return f("tanh")(used_once(), arg(0)(f("mul")(either_SQRT2RPI_add)));
    }

    /// x * (0.5? + 0.5 * tanh( sqrt(2/M_PI) * (x? + 0.044715 * x? ^ 3) ) )
    /// <item>? question mark means it doesn't explicitly match that item (anything will work)
    auto matcher_v0() const
    {
        auto mul_half_tanh = f("mul")(either_arg(0, 1)(has_value(0.5f), tanh_fn()));
        auto add_any_mul   = f("add")(any_arg(0, 1)(mul_half_tanh));
        return f("mul")(either_arg(0, 1)(any().bind("x"), add_any_mul));
    }

    /// x * 0.5 * (1.0 + tanh( sqrt(2/M_PI) * (x + 0.044715 * x ^ 3) ) )
    auto matcher_v1() const
    {
        auto add_one_tanh = f("add")(used_once(), either_arg(0, 1)(has_value(1.0), tanh_fn()));
        auto mul_half_x = f("mul")(used_once(), either_arg(0, 1)(has_value(0.5), any().bind("x")));
        return f("mul")(either_arg(0, 1)(mul_half_x, add_one_tanh));
    }
};
} // namespace detail

template <class F>
auto gelu_tanh(F f)
{
    return detail::gelu_tanh_matcher<F>{f}.matcher_v1();
}

inline auto gelu_tanh()
{
    return gelu_tanh([](auto x) { return name(x); });
}

} // namespace match
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MATCH_GELU_TANH_HPP
