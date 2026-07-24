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

// gpu::winograd_conv::compute_shape derives the output channel count K from the
// weight literal's shape, whose axis order depends on the store layout the host
// picked:
//   fp16 T-store [4, 3, K, C] and g-store [3, 3, K, C]  -> K at dim 2
//   fp32 full-U [4, 4, K, C] and S-store [3, 4, K, C]    -> K at dim 2
//   fp32 v-innermost [4, K, C, 4] (NHWC coalesced weight) -> K at dim 1
// The output takes the batch/spatial dims of the input and the winograd op's
// output_layout permutation. These tests exercise every (dtype, layout) pair with
// channel-expanding, channel-reducing, and square convs -- the v-inner regression
// (in_c != out_c returning in_c) was masked because every prior test was square.

using lens_t = std::vector<std::size_t>;

// Run compute_shape by placing the op in a module; conv->get_shape() is its result.
static migraphx::shape wino_shape(const migraphx::shape& x,
                                  const migraphx::shape& u,
                                  const std::vector<int64_t>& layout = {0, 1, 2, 3})
{
    migraphx::module m;
    auto xp   = m.add_parameter("x", x);
    auto up   = m.add_literal(migraphx::generate_literal(u));
    auto conv = m.add_instruction(
        migraphx::make_op("gpu::winograd_conv", {{"output_layout", layout}}), xp, up);
    return conv->get_shape();
}

// ---- fp32 full-U weight [4, 4, K, C] (K at dim 2) ----
TEST_CASE(fp32_full_u_expand)
{
    auto s = wino_shape({migraphx::shape::float_type, {1, 128, 16, 16}},
                        {migraphx::shape::float_type, {4, 4, 256, 128}});
    EXPECT(s.lens() == lens_t{1, 256, 16, 16});
    EXPECT(s.type() == migraphx::shape::float_type);
}

TEST_CASE(fp32_full_u_reduce)
{
    auto s = wino_shape({migraphx::shape::float_type, {1, 256, 16, 16}},
                        {migraphx::shape::float_type, {4, 4, 64, 256}});
    EXPECT(s.lens() == lens_t{1, 64, 16, 16});
}

TEST_CASE(fp32_full_u_square)
{
    auto s = wino_shape({migraphx::shape::float_type, {1, 192, 32, 32}},
                        {migraphx::shape::float_type, {4, 4, 192, 192}});
    EXPECT(s.lens() == lens_t{1, 192, 32, 32});
}

// ---- fp32 S-store weight [3, 4, K, C] (K at dim 2) ----
TEST_CASE(fp32_sstore_expand)
{
    auto s = wino_shape({migraphx::shape::float_type, {1, 128, 16, 16}},
                        {migraphx::shape::float_type, {3, 4, 256, 128}});
    EXPECT(s.lens() == lens_t{1, 256, 16, 16});
}

TEST_CASE(fp32_sstore_square)
{
    auto s = wino_shape({migraphx::shape::float_type, {1, 512, 8, 8}},
                        {migraphx::shape::float_type, {3, 4, 512, 512}});
    EXPECT(s.lens() == lens_t{1, 512, 8, 8});
}

// ---- fp32 v-innermost weight [4, K, C, 4] (K at dim 1) -- the regression ----
TEST_CASE(fp32_vinner_expand)
{
    auto s = wino_shape({migraphx::shape::float_type, {1, 128, 16, 16}},
                        {migraphx::shape::float_type, {4, 256, 128, 4}});
    EXPECT(s.lens() == lens_t{1, 256, 16, 16}); // K = 256, not C = 128
    EXPECT(s.type() == migraphx::shape::float_type);
}

TEST_CASE(fp32_vinner_reduce)
{
    auto s = wino_shape({migraphx::shape::float_type, {1, 256, 16, 16}},
                        {migraphx::shape::float_type, {4, 64, 256, 4}});
    EXPECT(s.lens() == lens_t{1, 64, 16, 16});
}

TEST_CASE(fp32_vinner_square)
{
    auto s = wino_shape({migraphx::shape::float_type, {1, 256, 64, 64}},
                        {migraphx::shape::float_type, {4, 256, 256, 4}});
    EXPECT(s.lens() == lens_t{1, 256, 64, 64});
}

// ---- fp16 T-store weight [4, 3, K, C] (K at dim 2; dim1 == 3 must not trip the
//      fp32-only v-inner test, which keys on dim1 != 4) ----
TEST_CASE(fp16_tstore_expand)
{
    auto s = wino_shape({migraphx::shape::half_type, {1, 128, 16, 16}},
                        {migraphx::shape::half_type, {4, 3, 256, 128}});
    EXPECT(s.lens() == lens_t{1, 256, 16, 16});
    EXPECT(s.type() == migraphx::shape::half_type);
}

TEST_CASE(fp16_tstore_reduce)
{
    auto s = wino_shape({migraphx::shape::half_type, {1, 512, 16, 16}},
                        {migraphx::shape::half_type, {4, 3, 128, 512}});
    EXPECT(s.lens() == lens_t{1, 128, 16, 16});
}

TEST_CASE(fp16_tstore_square)
{
    auto s = wino_shape({migraphx::shape::half_type, {1, 64, 128, 128}},
                        {migraphx::shape::half_type, {4, 3, 64, 64}});
    EXPECT(s.lens() == lens_t{1, 64, 128, 128});
}

// ---- fp16 g-store weight [3, 3, K, C] (K at dim 2) ----
TEST_CASE(fp16_gstore_expand)
{
    auto s = wino_shape({migraphx::shape::half_type, {1, 128, 16, 16}},
                        {migraphx::shape::half_type, {3, 3, 256, 128}});
    EXPECT(s.lens() == lens_t{1, 256, 16, 16});
    EXPECT(s.type() == migraphx::shape::half_type);
}

// ---- output_layout: NCHW is standard/packed; NHWC puts the channel axis innermost ----
TEST_CASE(nchw_output_layout)
{
    auto s = wino_shape({migraphx::shape::float_type, {1, 128, 16, 16}},
                        {migraphx::shape::float_type, {4, 4, 256, 128}},
                        {0, 1, 2, 3});
    EXPECT(s.lens() == lens_t{1, 256, 16, 16});
    EXPECT(s.standard());
}

TEST_CASE(nhwc_output_layout_fp32_vinner)
{
    auto s = wino_shape({migraphx::shape::float_type, {1, 128, 16, 16}},
                        {migraphx::shape::float_type, {4, 256, 128, 4}},
                        {0, 2, 3, 1});
    EXPECT(s.lens() == lens_t{1, 256, 16, 16});
    EXPECT(s.strides()[1] == 1); // channels-last
}

TEST_CASE(nhwc_output_layout_fp16)
{
    auto s = wino_shape({migraphx::shape::half_type, {1, 128, 16, 16}},
                        {migraphx::shape::half_type, {4, 3, 256, 128}},
                        {0, 2, 3, 1});
    EXPECT(s.lens() == lens_t{1, 256, 16, 16});
    EXPECT(s.strides()[1] == 1);
}

// ---- batch and spatial dims are carried from the input ----
TEST_CASE(batch_and_spatial_preserved)
{
    auto s = wino_shape({migraphx::shape::float_type, {4, 128, 30, 40}},
                        {migraphx::shape::float_type, {4, 256, 128, 4}});
    EXPECT(s.lens() == lens_t{4, 256, 30, 40});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
