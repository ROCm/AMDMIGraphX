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

#include <migraphx/config.hpp>
#include <cstdint>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void calculate_padding(int64_t idx,
                       std::vector<int64_t>& pads,
                       int64_t input_dim,
                       int64_t stride,
                       int64_t dilation,
                       int64_t weight_dim,
                       bool is_same_upper = true);

/*!
 * Calculate the padding for auto_padding. Used for dynamic shapes
 * where the padding calculation must be done at evaluation time.
 * \param tensor_lens input tensor image shape
 * \param k_lens weights kernel shape
 * \param strides strides for the kernel
 * \param dilations dilations for the kernel
 * \param use_upper put odd padding on upper or lower side
 * \return padding in the form of {x0_begin, x1_begin, ... x0_end , x1_end, ...}
 */
std::vector<std::size_t> calc_dyn_auto_pad(std::vector<std::size_t> tensor_lens,
                                           std::vector<std::size_t> k_lens,
                                           std::vector<std::size_t> strides,
                                           std::vector<std::size_t> dilations,
                                           bool use_upper = true);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
