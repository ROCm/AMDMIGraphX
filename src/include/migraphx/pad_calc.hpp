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
#ifndef MIGRAPHX_GUARD_OPERATORS_PAD_CALC_HPP
#define MIGRAPHX_GUARD_OPERATORS_PAD_CALC_HPP

#include <utility>
#include <cstdint>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

inline void calculate_padding(int64_t idx,
                              std::vector<int64_t>& pads,
                              int64_t input_dim,
                              int64_t stride,
                              int64_t dilation,
                              int64_t weight_dim,
                              bool is_same_upper = true)
{
    int64_t output_dim     = (input_dim + stride - 1) / stride; // round up result
    int64_t new_weight_dim = weight_dim + (weight_dim - 1) * (dilation - 1);
    int64_t pad =
        std::max(static_cast<int64_t>(0), (output_dim - 1) * stride + new_weight_dim - input_dim);
    auto pad_ndims = pads.size() / 2;

    if(is_same_upper)
    {
        pads[idx]             = pad / 2;
        pads[idx + pad_ndims] = pad - pad / 2;
    }
    else
    {
        pads[idx + pad_ndims] = pad / 2;
        pads[idx]             = pad - pad / 2;
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
