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
        const auto& w_shape = w->get_shape();

        // Reduce over all weight axes except output channel (axis 0).
        std::vector<std::int64_t> reduce_axes;
        for(std::size_t i = 1; i < w_shape.ndim(); ++i)
            reduce_axes.push_back(static_cast<std::int64_t>(i));

        auto abs_w = m.insert_instruction(ins, make_op("abs"), w);
        auto scale_max =
            m.insert_instruction(ins, make_op("reduce_max", {{"axes", reduce_axes}}), abs_w);

        // Clamp scale away from zero to keep div well-defined when a channel is all zeros.
        auto eps    = m.add_literal(literal{shape{out_type}, {1e-30f}});
        auto eps_bc = m.insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", scale_max->get_shape().lens()}}), eps);
        auto scale_kd = m.insert_instruction(ins, make_op("max"), scale_max, eps_bc);

        // Drop the reduced singleton axes to get a 1-D per-output-channel scale.
        auto scale =
            m.insert_instruction(ins, make_op("squeeze", {{"axes", reduce_axes}}), scale_kd);

        // Broadcast 1-D scale along axis 0 of W ([oc, ic, kh, kw]).
        auto scale_w_bc = m.insert_instruction(
            ins, make_op("broadcast", {{"axis", 0}, {"out_lens", w_shape.lens()}}), scale);
        auto w_scaled = m.insert_instruction(ins, make_op("div"), w, scale_w_bc);

        auto x_h =
            m.insert_instruction(ins, make_op("convert", {{"target_type", shape::half_type}}), x);
        auto w_h = m.insert_instruction(
            ins, make_op("convert", {{"target_type", shape::half_type}}), w_scaled);

        auto half_conv = m.insert_instruction(ins, ins->get_operator(), x_h, w_h);
        auto converted =
            m.insert_instruction(ins, make_op("convert", {{"target_type", out_type}}), half_conv);

        // Broadcast 1-D scale along channel axis 1 of the conv output ([N, oc, ...]).
        auto scale_out_bc = m.insert_instruction(
            ins, make_op("broadcast", {{"axis", 1}, {"out_lens", out_shape.lens()}}), scale);
        auto result = m.insert_instruction(ins, make_op("mul"), converted, scale_out_bc);

        m.replace_instruction(ins, result);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
