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

TEST_CASE(qlinearconv_perchannel_weightbias_test)
{
    migraphx::program p = read_onnx("qlinearconv_perchannel_weightbias_test.onnx");

    p.compile(migraphx::make_target("ref"));

    // Generate data for input X
    migraphx::shape sx{migraphx::shape::uint8_type, {1, 3, 224, 224}};
    std::vector<uint8_t> x_data(sx.elements());
    for(std::size_t i = 0; i < x_data.size(); ++i)
    {
        x_data[i] = static_cast<uint8_t>(i % 256);
    }

    migraphx::parameter_map pp;
    pp["X"] = migraphx::argument(sx, x_data.data());

    auto result = p.eval(pp).back();

    // Verify output shape
    EXPECT(result.get_shape().lens() == std::vector<std::size_t>{1, 64, 112, 112});
    EXPECT(result.get_shape().type() == migraphx::shape::uint8_type);

    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Golden computed from numpy reference
    std::vector<uint8_t> gold_prefix = {129, 134, 133, 133, 133, 133, 133, 133, 133, 133, 133,
                                        133, 133, 133, 133, 139, 132, 127, 125, 125, 125, 125,
                                        125, 125, 125, 125, 125, 125, 125, 125, 125, 127};

    std::vector<uint8_t> result_prefix(result_vector.begin(), result_vector.begin() + 32);
    EXPECT(migraphx::verify::verify_rms_range(result_prefix, gold_prefix));
}
