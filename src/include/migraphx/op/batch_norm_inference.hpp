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
#ifndef MIGRAPHX_GUARD_OPERATORS_BATCH_NORM_HPP
#define MIGRAPHX_GUARD_OPERATORS_BATCH_NORM_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <cmath>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct batch_norm_inference
{
    float epsilon  = 1.0e-6f;
    float momentum = 0.9f;

    std::string name() const { return "batch_norm_inference"; }

    enum bn_infer_mode_t
    {
        per_activation,
        spatial,
    };

    bn_infer_mode_t bn_mode = spatial;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(
            f(self.epsilon, "epsilon"), f(self.momentum, "momentum"), f(self.bn_mode, "bn_mode"));
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(5);
        check_shapes{inputs.data(), inputs.data() + 1, *this}.same_ndims();
        check_shapes{inputs.data() + 1, inputs.data() + inputs.size(), *this}.same_shape();
        return inputs.front();
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
