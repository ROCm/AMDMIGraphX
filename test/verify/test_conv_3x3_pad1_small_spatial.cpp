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

// Regression for the F(2,3) winograd boundary-tile load. The X buffer
// descriptor only enforces [0, byte_count), so h-OOB / w-OOB loads silently
// wrap into the neighbouring channel/row instead of returning zero. Small
// spatial sizes have a large fraction of boundary tiles, so any miscount in
// the V transform leaks into the first output.
template <std::size_t H, std::size_t W, std::size_t K = 64, std::size_t C = 64>
struct test_conv_3x3_pad1_small_spatial
    : verify_program<test_conv_3x3_pad1_small_spatial<H, W, K, C>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::half_type, {1, C, H, W}});
        // Winograd matcher requires can_eval() on weights — add as literal.
        auto w = mm->add_literal(
            migraphx::generate_literal({migraphx::shape::half_type, {K, C, 3, 3}}, 1));
        mm->add_instruction(migraphx::make_op("convolution",
                                              {{"padding", {1, 1}},
                                               {"stride", {1, 1}},
                                               {"dilation", {1, 1}}}),
                            x,
                            w);
        return p;
    }
    std::string section() const { return "conv"; }
};

// 6x6: every tile is a border tile (h0=-1, w0=-1 corner; last tile at
// h0=3, w0=3 has h+3=6, w+3=6 wrap into next channel).
template struct test_conv_3x3_pad1_small_spatial<6, 6>;
// 8x8: even spatial, last tile in each direction is right at the H/W edge.
template struct test_conv_3x3_pad1_small_spatial<8, 8>;
// 12x12: typical bottleneck spatial in the topaz model; mixes interior and
// border tiles within the same workgroup.
template struct test_conv_3x3_pad1_small_spatial<12, 12>;
