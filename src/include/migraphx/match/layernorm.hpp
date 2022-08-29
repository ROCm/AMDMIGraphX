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
#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_MATCH_LAYERNORM_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_MATCH_LAYERNORM_HPP

#include <migraphx/config.hpp>
#include <migraphx/matcher.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace match {

namespace detail {
template <class F>
struct layernorm_matcher
{
    F f;
    auto x_minus_mean() const
    {
        return f("sub")(arg(0)(any().bind("x")), arg(1)(skip_broadcasts(f("reduce_mean"))));
    }

    auto variance() const
    {
        return f("reduce_mean")(arg(0)(f("pow")(arg(0)(x_minus_mean()), arg(1)(has_value(2.0f)))));
    }

    auto layernorm_onnx() const
    {
        return f("div")(arg(0)(x_minus_mean()),

                        arg(1)(skip_broadcasts(f("sqrt")(arg(0)(
                            f("add")(either_arg(0, 1)(variance(), is_constant().bind("eps"))))))));
    }

    auto matcher() const { return layernorm_onnx(); }
};
} // namespace detail

template <class F>
auto layernorm(F f)
{
    return detail::layernorm_matcher<F>{f}.matcher();
}

inline auto layernorm()
{
    return layernorm([](auto x) { return name(x); });
}

} // namespace match
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
