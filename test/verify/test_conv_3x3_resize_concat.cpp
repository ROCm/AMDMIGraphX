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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

// The u-net decoder level: a 2x bilinear upsample concatenated on channels
// with a full-resolution skip tensor, feeding a 3x3/pad-1 conv + bias +
// leaky_relu. The winograd resize matcher fuses the upsample into the conv's
// input load, reading the low-resolution source for the first Ca channels and
// the skip tensor for the rest (Ca must be a multiple of 16, the kernel's
// channel block). The fused pointwise exercises the writeback with extra
// inputs. The ragged tail (Cb % 8 != 0) and K tail (K % 8 != 0) cover the
// partial-block masking; batch 2 covers the n-offset addressing.
template <std::size_t Ca, std::size_t Cb, std::size_t K, std::size_t H, std::size_t W>
struct test_conv_3x3_resize_concat : verify_program<test_conv_3x3_resize_concat<Ca, Cb, K, H, W>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto a   = mm->add_parameter("a", {migraphx::shape::half_type, {2, Ca, H, W}});
        auto b   = mm->add_parameter("b", {migraphx::shape::half_type, {2, Cb, 2 * H, 2 * W}});
        // Winograd matcher requires can_eval() on weights -- add as literals.
        // The leading convs keep the resize/concat inside the channels-last
        // region layout_convolution creates (fed straight from parameters they
        // stay NCHW and the fusion does not apply), matching the u-net
        // decoder structure.
        auto wa = mm->add_literal(
            migraphx::generate_literal({migraphx::shape::half_type, {Ca, Ca, 3, 3}}, 3));
        auto wb = mm->add_literal(
            migraphx::generate_literal({migraphx::shape::half_type, {Cb, Cb, 3, 3}}, 4));
        auto w = mm->add_literal(
            migraphx::generate_literal({migraphx::shape::half_type, {K, Ca + Cb, 3, 3}}, 1));
        auto bias =
            mm->add_literal(migraphx::generate_literal({migraphx::shape::half_type, {K}}, 2));
        auto ca = mm->add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
            a,
            wa);
        auto cb = mm->add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
            b,
            wb);
        auto r = mm->add_instruction(
            migraphx::make_op("resize",
                              {{"sizes", {2, Ca, 2 * H, 2 * W}},
                               {"mode", "linear"},
                               {"coordinate_transformation_mode", "asymmetric"}}),
            ca);
        auto cat  = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), r, cb);
        auto conv = mm->add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
            cat,
            w);
        auto bias_b = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv->get_shape().lens()}}),
            bias);
        auto add = mm->add_instruction(migraphx::make_op("add"), conv, bias_b);
        mm->add_instruction(migraphx::make_op("leaky_relu", {{"alpha", 0.2}}), add);
        return p;
    }
    std::string section() const { return "conv"; }
};

template struct test_conv_3x3_resize_concat<16, 24, 17, 8, 8>;
template struct test_conv_3x3_resize_concat<32, 16, 32, 12, 10>;
