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
#include <migraphx/fast_mm.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void fast_mm::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "convolution")
            continue;

        const auto out_type = ins->get_shape().type();
        if(out_type != shape::float_type)
            continue;

        const auto& out_shape = ins->get_shape();
        if(out_shape.dynamic())
            continue;

        auto inputs = ins->inputs();
        auto x      = inputs[0];
        auto w      = inputs[1];
        if(not w->can_eval())
            continue;

        // The hi/lo split below assumes a single input-channel group.
        auto op_val = ins->get_operator().to_value();
        if(op_val.contains("group") and op_val.at("group").to<int>() != 1)
            continue;

        const auto& w_shape   = w->get_shape();
        std::size_t reduction = 1;
        for(std::size_t i = 1; i < w_shape.ndim(); ++i)
            reduction *= w_shape.lens()[i];
        // Skip if reduction compounds fp16 input rounding beyond what the
        // verify-test 80-ULP fp32 tolerance can absorb.
        if(reduction > 512)
            continue;

        // Skip when the conv is too small to benefit from fp16 anyway. Tiny
        // convs also tend to follow upstream reductions, which produce small
        // values whose fp16 rounding becomes the dominant absolute error.
        std::size_t total_ops = out_shape.elements() * reduction;
        if(total_ops < 1024)
            continue;

        auto x_h =
            m.insert_instruction(ins, make_op("convert", {{"target_type", shape::half_type}}), x);
        auto w_h =
            m.insert_instruction(ins, make_op("convert", {{"target_type", shape::half_type}}), w);

        auto half_conv = m.insert_instruction(ins, ins->get_operator(), x_h, w_h);
        auto converted =
            m.insert_instruction(ins, make_op("convert", {{"target_type", out_type}}), half_conv);

        m.replace_instruction(ins, converted);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
