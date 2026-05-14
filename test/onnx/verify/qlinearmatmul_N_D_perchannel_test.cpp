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

TEST_CASE(qlinearmatmul_n_d_perchannel_test)
{
    migraphx::program p = read_onnx("qlinearmatmul_N_D_perchannel_test.onnx");

    p.compile(migraphx::make_target("ref"));

    // Generate test data for A
    migraphx::shape a_shape{migraphx::shape::uint8_type, {2, 3, 4}};
    std::vector<uint8_t> data_a = {254, 109, 126, 66,  220, 98, 230, 17, 83,  106, 123, 57,
                                   214, 225, 96,  113, 126, 47, 73,  32, 174, 224, 111, 153};

    migraphx::parameter_map pp;
    pp["A"] = migraphx::argument(a_shape, data_a.data());

    auto result = p.eval(pp).back();

    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Verify output shape
    EXPECT(result.get_shape().lens() == std::vector<std::size_t>{2, 3, 5});
    EXPECT(result.get_shape().type() == migraphx::shape::uint8_type);

    // Gold values computed using numpy reference implementation
    std::vector<uint8_t> gold = {119, 142, 138, 130, 134, 143, 133, 131, 127, 136,
                                 144, 130, 127, 129, 128, 124, 145, 124, 133, 132,
                                 132, 142, 90,  129, 125, 123, 133, 143, 131, 132};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
