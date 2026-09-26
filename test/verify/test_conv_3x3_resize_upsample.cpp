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
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

// A 2x bilinear upsample (ONNX Resize mode=linear,
// coordinate_transformation_mode=asymmetric) feeding a 3x3/pad-1 convolution:
// the pattern the F(2,3) winograd matcher fuses into the conv's input load
// (find_winograd_f23_resize), reading the low-resolution source directly. On
// non-gfx12 targets this validates the same graph through the default
// lowering. The 4x4 source hits the all-border-tile and source-edge-clamp
// paths; the 24->17 case exercises partial channel blocks (C % 16 != 0) and
// the K tail (K % 8 != 0).
template <std::size_t C, std::size_t K, std::size_t H, std::size_t W>
struct test_conv_3x3_resize_upsample : verify_program<test_conv_3x3_resize_upsample<C, K, H, W>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::half_type, {1, C, H, W}});
        // Winograd matcher requires can_eval() on weights -- add as literals.
        // The leading conv keeps the resize inside the channels-last region
        // layout_convolution creates (a resize fed straight from a parameter
        // stays NCHW and the fusion does not apply), matching the u-net
        // decoder structure.
        auto w0 = mm->add_literal(
            migraphx::generate_literal({migraphx::shape::half_type, {C, C, 3, 3}}, 3));
        auto w = mm->add_literal(
            migraphx::generate_literal({migraphx::shape::half_type, {K, C, 3, 3}}, 1));
        auto c0 = mm->add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
            x,
            w0);
        auto r = mm->add_instruction(
            migraphx::make_op("resize",
                              {{"sizes", {1, C, 2 * H, 2 * W}},
                               {"mode", "linear"},
                               {"coordinate_transformation_mode", "asymmetric"}}),
            c0);
        mm->add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
            r,
            w);
        return p;
    }
    std::string section() const { return "conv"; }
};

template struct test_conv_3x3_resize_upsample<32, 32, 16, 16>;
template struct test_conv_3x3_resize_upsample<24, 17, 4, 4>;
