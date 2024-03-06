/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_MATCH_RMSNORM_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_MATCH_RMSNORM_HPP

#include <migraphx/config.hpp>
#include <migraphx/matcher.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace match {

namespace detail {
template <class F>
struct rmsnorm_matcher
{
    F f;

    auto last_axis() const
    {
        return make_basic_pred_matcher([](instruction_ref ins) {
            auto v = ins->get_operator().to_value();
            if(not v.contains("axes"))
                return false;
            auto axes = v["axes"].to_vector<std::size_t>();
            if(axes.size() != 1)
                return false;
            return (axes.front() == -1) or
                   (axes.front() == ins->inputs().front()->get_shape().lens().size() - 1);
        });
    }

    auto reduce_op() const
    {
        return any(any_of(f("reduce_mean")(last_axis()), f("reduce_sum")(last_axis())));
    }

    auto variance() const
    {
        return reduce_op()(
                   arg(0)(f("pow")(arg(0)(any().bind("x")), arg(1)(has_value(2.0f)).bind("pow"))))
            .bind("reduce_op");
    }

    auto sqrt_add_eps(const std::string& name) const
    {
        auto add_eps =
            f("add")(either_arg(0, 1)(variance(), is_constant().bind("eps"))).bind("add");
        auto sqrt = f(name)(arg(0)(any_of(add_eps, variance()))).bind("sqrt");
        return any_of(sqrt, f("multibroadcast")(arg(0)(sqrt)).bind("mbcast"));
    }

    auto rmsnorm() const
    {
        auto div_sqrt  = f("div")(arg(0)(any()), arg(1)(sqrt_add_eps("sqrt"))).bind("div");
        auto mul_rsqrt = f("mul")(either_arg(0, 1)(any(), sqrt_add_eps("rsqrt"))).bind("mul");
        return any(any_of(div_sqrt, mul_rsqrt).bind("div_mul"));
    }

    auto matcher() const { return rmsnorm(); }
};
} // namespace detail

template <class F>
auto rmsnorm(F f)
{
    return detail::rmsnorm_matcher<F>{f}.matcher();
}

inline auto rmsnorm()
{
    return rmsnorm([](auto x) { return name(x); });
}

} // namespace match
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
