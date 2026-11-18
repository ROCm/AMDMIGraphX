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

#include <numeric>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

// data {2, 5} = {0, 1, ..., 9}; indices {2, k} pick along axis 1, so out[r][c] = 5 * r + idx
static void check_gather_elements(migraphx::program p, std::vector<int64_t> indices)
{
    p.compile(migraphx::make_target("ref"));
    std::vector<float> data(10);
    std::iota(data.begin(), data.end(), 0);
    auto k = indices.size() / 2;
    migraphx::parameter_map pp;
    pp["data"]    = migraphx::argument({migraphx::shape::float_type, {2, 5}}, data.data());
    pp["indices"] = migraphx::argument({migraphx::shape::int64_type, {2, k}}, indices.data());
    auto result   = p.eval(pp).back();
    EXPECT(result.get_shape().lens() == std::vector<std::size_t>{2, k});
    std::vector<float> gold;
    for(std::size_t i = 0; i < indices.size(); i++)
        gold.push_back(5 * (i / k) + indices[i]);
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    EXPECT(result_vector == gold);
}

TEST_CASE(gather_elements_dyn_axis_test)
{
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    auto p                        = read_onnx("gather_elements_dyn_axis_test.onnx", options);
    check_gather_elements(p, {4, 0});
    check_gather_elements(p, {1, 3, 0, 2, 4, 4});
}

TEST_CASE(gather_elements_symbolic_axis_test)
{
    migraphx::onnx_options options;
    options.use_symbolic_shapes   = true;
    options.default_dyn_dim_value = {1, 4};
    auto p                        = read_onnx("gather_elements_dyn_axis_test.onnx", options);
    EXPECT(p.get_output_shapes().back().symbolic());
    check_gather_elements(p, {4, 0});
    check_gather_elements(p, {1, 3, 0, 2, 4, 4});
}
