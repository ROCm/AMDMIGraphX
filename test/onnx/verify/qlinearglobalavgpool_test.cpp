/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

TEST_CASE(qlinearglobalavgpool_test)
{
    // github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md
    // #com.microsoft.QLinearGlobalAveragePool

    migraphx::program p = read_onnx("qlinearglobalavgpool_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape sh_x{migraphx::shape::uint8_type, {1, 3, 4, 4}};
    std::vector<uint8_t> data_x = {160, 156, 152, 148, 144, 140, 136, 132, 124, 120, 116, 112,
                                   108, 104, 100, 96,  64,  72,  80,  88,  96,  104, 112, 120,
                                   136, 144, 152, 160, 168, 176, 184, 192, 120, 121, 122, 123,
                                   124, 125, 126, 127, 129, 130, 131, 132, 133, 134, 135, 136};

    migraphx::parameter_map pp;
    pp["X"] = migraphx::argument(sh_x, data_x.data());

    auto result = p.eval(pp).back();

    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<uint8_t> gold = {64, 64, 64};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
