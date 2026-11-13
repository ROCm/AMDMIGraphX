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
#include <migraphx/eliminate_pad.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/op/pad.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/float_equal.hpp>
#include <algorithm>
#include <functional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// Op padding can only absorb non-negative pads on the spatial dims
static bool can_fold_pads(const op::pad& pad_op)
{
    const auto& pads = pad_op.pads;
    auto ndim        = pad_op.pad_ndims();
    if(std::any_of(pads.begin(), pads.end(), [](auto p) { return p < 0; }))
        return false;
    return pads[0] == 0 and pads[1] == 0 and pads[ndim] == 0 and pads[ndim + 1] == 0;
}

// Pads of the spatial dims as {begin..., end...}
static std::vector<std::size_t> spatial_pads(const op::pad& pad_op)
{
    const auto& pads = pad_op.pads;
    auto ndim        = pad_op.pad_ndims();
    std::vector<std::size_t> result(pads.begin() + 2, pads.begin() + ndim);
    result.insert(result.end(), pads.begin() + ndim + 2, pads.end());
    return result;
}

// Expand a symmetric {p...} op padding to the {begin..., end...} form
static std::vector<std::size_t> expand_padding(const std::vector<std::size_t>& padding,
                                               std::size_t kdims)
{
    if(padding.size() == 2 * kdims)
        return padding;
    std::vector<std::size_t> result(padding);
    result.insert(result.end(), padding.begin(), padding.end());
    return result;
}

void eliminate_pad::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        const std::string& op_name = ins->name();
        if(op_name != "convolution" and op_name != "im2col" and op_name != "pooling")
            continue;
        auto input = ins->inputs().front();
        if(input->name() != "pad")
            continue;
        auto pad_op = any_cast<op::pad>(input->get_operator());
        // Only support folding zero padding into convolution/im2col/pooling
        if(pad_op.mode != op::pad::pad_op_mode_t::constant_pad or
           not float_equal(pad_op.value, 0.0f))
            continue;
        if(not can_fold_pads(pad_op))
            continue;
        if(not asym_pad and not pad_op.symmetric())
            continue;
        auto op = ins->get_operator();
        auto v  = op.to_value();
        // Auto padding is computed from dynamic input dims at runtime, which the pad would change
        bool auto_pad = v.at("padding_mode").to<int>() != op::padding_mode_t::default_;
        if(auto_pad and input->get_shape().dynamic())
            continue;
        // Average pooling may exclude its padding from the divisor, unlike a pad op
        if(op_name == "pooling" and any_cast<op::pooling>(op).mode == op::pooling_mode::average)
            continue;

        auto kdims   = pad_op.pad_ndims() - 2;
        auto padding = expand_padding(v.at("padding").to_vector<std::size_t>(), kdims);
        auto pads    = spatial_pads(pad_op);
        std::transform(pads.begin(), pads.end(), padding.begin(), padding.begin(), std::plus<>{});
        // Static shapes ignore padding_mode, so the explicit padding now describes the op
        op.from_value({{"padding", padding}, {"padding_mode", op::padding_mode_t::default_}});

        auto new_inputs    = ins->inputs();
        new_inputs.front() = input->inputs().front();
        m.replace_instruction(ins, op, new_inputs);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
