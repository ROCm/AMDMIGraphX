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

// y[n] = sum_k x[k] * w[k][n] * s[k / 16][n] * s2
static std::vector<float> run_nvfp4_dq_matmul(const std::string& model)
{
    migraphx::program p = read_onnx(model);
    p.compile(migraphx::make_target("ref"));

    // Distinct inputs for the two quantization blocks along K
    std::vector<float> x(16, 1.0f);
    x.resize(32, 2.0f);
    migraphx::parameter_map pm;
    pm["x"] = migraphx::argument{{migraphx::shape::float_type, {1, 32}}, x.data()};

    auto result = p.eval(pm).back();
    std::vector<float> rv;
    result.visit([&](auto output) { rv.assign(output.begin(), output.end()); });
    return rv;
}

TEST_CASE(nvfp4_dq_matmul_test)
{
    std::vector<float> gold = {16.0f, -240.0f};
    EXPECT(run_nvfp4_dq_matmul("nvfp4_dq_matmul_test.onnx") == gold);
}

TEST_CASE(nvfp4_dq_matmul_int32_data_test)
{
    std::vector<float> gold = {16.0f, -240.0f};
    EXPECT(run_nvfp4_dq_matmul("nvfp4_dq_matmul_int32_data_test.onnx") == gold);
}
