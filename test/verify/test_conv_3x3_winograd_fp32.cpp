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

// fp32 F(2,3) winograd (the FMA/DPP kernel, gated on MIGRAPHX_ENABLE_WINOGRAD).
// Odd spatial size exercises the boundary tiles (halo padding); the channel
// count is not a multiple of the per-lane output block so the partial-KO store
// path is covered. Without the env var this validates the default lowering.
struct test_conv_3x3_winograd_fp32 : verify_program<test_conv_3x3_winograd_fp32>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {1, 20, 15, 15}});
        // Winograd matcher requires can_eval() on weights -> add as a literal.
        auto w = mm->add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {36, 20, 3, 3}}, 1));
        mm->add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
            x,
            w);
        return p;
    }
    std::string section() const { return "conv"; }
};
