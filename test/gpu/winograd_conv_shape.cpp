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

#include <test.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/gpu/target.hpp> // links migraphx_gpu so gpu::winograd_conv is registered

// gpu::winograd_conv derives its output channel count K from the weight literal's
// shape, whose axis order differs per store layout:
//   fp16 [4|3, 3, K, C] and fp32 full-U / S-store [4|3, 4, K, C]  -> K at dim 2
//   fp32 v-innermost [4, K, C, 4] (NHWC coalesced weight load)    -> K at dim 1
// Regression: with in_c != out_c the v-inner layout used to return in_c (dim 2)
// as the output channel count, so a downstream op's shape check failed (topaz
// gfrf-v2-fp32 NHWC compile). Every prior winograd test was square (in_c == out_c),
// which masked it, so these all use a non-square 128 -> 256 conv.
static migraphx::shape winograd_out_shape(const migraphx::shape& x, const migraphx::shape& u)
{
    migraphx::module m;
    auto xp   = m.add_parameter("x", x);
    auto up   = m.add_literal(migraphx::generate_literal(u));
    auto conv = m.add_instruction(migraphx::make_op("gpu::winograd_conv"), xp, up);
    return conv->get_shape();
}

TEST_CASE(winograd_conv_vinner_non_square)
{
    // fp32 v-inner weight [u=4, K=256, C=128, v=4]: K is at dim 1.
    auto s = winograd_out_shape({migraphx::shape::float_type, {1, 128, 16, 16}},
                                {migraphx::shape::float_type, {4, 256, 128, 4}});
    EXPECT(s.lens() == std::vector<std::size_t>{1, 256, 16, 16});
}

TEST_CASE(winograd_conv_full_u_non_square)
{
    // fp32 full-U weight [u=4, v=4, K=256, C=128]: K is at dim 2.
    auto s = winograd_out_shape({migraphx::shape::float_type, {1, 128, 16, 16}},
                                {migraphx::shape::float_type, {4, 4, 256, 128}});
    EXPECT(s.lens() == std::vector<std::size_t>{1, 256, 16, 16});
}

TEST_CASE(winograd_conv_sstore_non_square)
{
    // fp32 S-store weight [i=3, v=4, K=256, C=128]: K is at dim 2.
    auto s = winograd_out_shape({migraphx::shape::float_type, {1, 128, 16, 16}},
                                {migraphx::shape::float_type, {3, 4, 256, 128}});
    EXPECT(s.lens() == std::vector<std::size_t>{1, 256, 16, 16});
}

TEST_CASE(winograd_conv_fp16_non_square)
{
    // fp16 weight [4, 3, K=256, C=128]: K at dim 2; the fp32-only v-inner test
    // (dim1 != 4, here dim1 == 3) must not misfire and return C.
    auto s = winograd_out_shape({migraphx::shape::half_type, {1, 128, 16, 16}},
                                {migraphx::shape::half_type, {4, 3, 256, 128}});
    EXPECT(s.lens() == std::vector<std::size_t>{1, 256, 16, 16});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
