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

#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(resize_outsize_nondivisible_test)
{
    // Resize using int64 sizes where sizes/input produces non-integer scales (5/3).
    // Verifies the output shape is exactly [1,1,5,5] and the values are correct.
    migraphx::program p = read_onnx("resize_outsize_nondivisible_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 3, 3}};
    std::vector<float> dx(sx.elements());
    std::iota(dx.begin(), dx.end(), 0.1f);

    migraphx::parameter_map pp;
    pp["X"] = migraphx::argument(sx, dx.data());

    auto result = p.eval(pp).back();

    // Verify the output shape is exactly [1,1,5,5], not [1,1,4,4]
    std::vector<std::size_t> expected_lens = {1, 1, 5, 5};
    EXPECT(result.get_shape().lens() == expected_lens);

    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // nearest mode=floor, coordinate_transformation_mode=asymmetric
    // Input 3x3: {{0.1, 1.1, 2.1}, {3.1, 4.1, 5.1}, {6.1, 7.1, 8.1}}
    // scale = 5/3, idx_op: out_idx / scale
    // For each output index 0..4: floor(i / (5/3)) = floor(i * 3/5) -> {0,0,1,1,2}
    // clang-format off
    std::vector<float> gold = {0.1f, 0.1f, 1.1f, 1.1f, 2.1f,
                               0.1f, 0.1f, 1.1f, 1.1f, 2.1f,
                               3.1f, 3.1f, 4.1f, 4.1f, 5.1f,
                               3.1f, 3.1f, 4.1f, 4.1f, 5.1f,
                               6.1f, 6.1f, 7.1f, 7.1f, 8.1f};
    // clang-format on

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
