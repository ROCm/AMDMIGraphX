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
#ifndef MIGRAPHX_GUARD_KERNELS_SOFTMAX_HPP
#define MIGRAPHX_GUARD_KERNELS_SOFTMAX_HPP

#include <migraphx/kernels/reduce.hpp>
#include <migraphx/kernels/ops.hpp>

namespace migraphx {

template <index_int Axis, class Input, class Output>
__device__ void softmax(Input input, Output output)
{
    reduce::block::run<reduce::with_axis<Input, Axis>>([&](auto, auto r) {
        auto batch_max = r.reduce(op::max{}, lowest{}, op::id{})(input);
        auto batch_sum =
            r.reduce(op::sum{}, 0, [&](auto x) { return migraphx::exp(x - batch_max); })(input);
        r.inner([&](auto& y, auto x) { y = migraphx::exp(x - batch_max) / batch_sum; })(output,
                                                                                        input);
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_SOFTMAX_HPP
