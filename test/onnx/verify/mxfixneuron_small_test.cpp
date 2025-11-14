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

TEST_CASE(mxfixneuron_small_test)
{
    migraphx::program p = read_onnx("mxfixneuron_small_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<std::size_t> input_lens{4, 4};
    auto input_type = migraphx::shape::float_type;
    migraphx::shape data_shape{input_type, input_lens};
    std::vector<float> data = {-100.f,
                               -12.f,
                               32.f,
                               819.f,
                               -6.f,
                               -5.75f,
                               -5.50f,
                               -5.25f,
                               -5.f,
                               -0.30f,
                               -1.40f,
                               -1.20f,
                               2.0f,
                               0.25f,
                               0.33f,
                               2.20f};
    migraphx::parameter_map pp;
    pp["input"] = migraphx::argument(data_shape, data.data());
    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // hand calculated values
    std::vector<float> gold = {-128.0,
                               -0.0,
                               0.0,
                               768.0,
                               -6.0,
                               -6.0,
                               -6.0,
                               -6.0,
                               -4.0,
                               -0.5,
                               -1.5,
                               -1.0,
                               2.0,
                               0.25,
                               0.25,
                               2.0};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
