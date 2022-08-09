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
#ifndef MIGRAPHX_GUARD_KERNELS_LAYERNORM_HPP
#define MIGRAPHX_GUARD_KERNELS_LAYERNORM_HPP
#include <migraphx/kernels/reduce.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/print.hpp>

namespace migraphx {

template <index_int Axis,
          class F,
          class BinOp,
          class Output,
          class Input1,
          class Input2,
          class... Inputs>
__device__ void generic_binary_layernorm(
    F compute, BinOp op, Output output, Input1 input1, Input2 input2, Inputs... inputs)
{
    using reduce_output = reduce::with_axis<Input1, Axis>;
    constexpr auto relements =
        get_shape_c<Input1>{}.elements() / get_shape_c<reduce_output>{}.elements();
    MIGRAPHX_ASSERT(relements > 0);
    reduce::block::run<reduce_output>([&](auto, auto r) {
        using value_type = typename Input1::type;
        auto mean        = [&](auto f) {
            return r.reduce(op::sum{}, 0, [&](auto x1, auto x2) {
                return f(x1, x2) / value_type{relements};
            })(input1, input2);
        };
        // mean(x)
        auto mean_x = mean(op);
        // mean(m ^ 2)
        auto mean_m2 = mean([&](auto x1, auto x2) {
            auto m = op(x1, x2) - mean_x;
            return m * m;
        });

        r.inner([&](auto& y, auto x1, auto x2, auto... xs) {
            auto m = op(x1, x2) - mean_x;
            // m * rsqrt(mean(m ^ 2) + 1e-12)
            y = compute(m * rsqrt(mean_m2 + value_type{1e-12}), xs...);
        })(output, input1, input2, inputs...);
    });
}

template <index_int Axis, class F, class Output, class Input, class... Inputs>
__device__ void layernorm(F compute, Output output, Input input, Inputs... inputs)
{
    generic_binary_layernorm<Axis>(
        compute, [](auto x, auto) { return x; }, output, input, input, inputs...);
}

template <index_int Axis, class F, class Output, class Input1, class Input2, class... Inputs>
__device__ void
add_layernorm(F compute, Output output, Input1 input1, Input2 input2, Inputs... inputs)
{
    generic_binary_layernorm<Axis>(
        compute, [](auto x1, auto x2) { return x1 + x2; }, output, input1, input2, inputs...);
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_LAYERNORM_HPP
