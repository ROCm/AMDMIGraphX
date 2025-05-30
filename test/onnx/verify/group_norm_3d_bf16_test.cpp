/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <onnx_verify_utils.hpp>

TEST_CASE(group_norm_bf16_test)
{
    using migraphx::bf16;
    std::vector<bf16> scale{bf16{1.2}, bf16{0.8}, bf16{0.4}, bf16{1.6}};
    std::vector<bf16> bias{bf16{0.5}, bf16{0.2}, bf16{0.9}, bf16{0.4}};
    std::vector<bf16> result_vector =
        norm_test<bf16>({1, 4, 2}, scale, bias, read_onnx("group_norm_3d_bf16_test.onnx"));
    std::vector<bf16> gold = {bf16{-0.96093750},
                              bf16{-0.37500000},
                              bf16{0.78125000},
                              bf16{1.17187500},
                              bf16{0.41210938},
                              bf16{0.60546875},
                              bf16{1.56250000},
                              bf16{2.34375000}};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
