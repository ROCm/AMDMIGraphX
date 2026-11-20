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

// Expand x = {1, 2, 3, ...} of shape {n, 1, 1}; each element is repeated 16 times
static void check_expand(migraphx::program p, std::size_t n, std::vector<std::size_t> out_lens)
{
    p.compile(migraphx::make_target("ref"));
    std::vector<float> x(n);
    std::iota(x.begin(), x.end(), 1);
    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument({migraphx::shape::float_type, {n, 1, 1}}, x.data());
    auto result = p.eval(pp).back();
    EXPECT(result.get_shape().lens() == out_lens);
    std::vector<float> gold;
    for(std::size_t i = 0; i < out_lens.front(); i++)
        gold.insert(gold.end(), 16, x[i % n]);
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    EXPECT(result_vector == gold);
}

TEST_CASE(expand_dyn_input_static_dims_test)
{
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 3};
    auto p                        = read_onnx("expand_dyn_input_static_dims_test.onnx", options);
    check_expand(p, 3, {3, 4, 4});
    check_expand(p, 1, {3, 4, 4});
}

TEST_CASE(expand_dyn_input_ones_dims_test)
{
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 3};
    auto p                        = read_onnx("expand_dyn_input_ones_dims_test.onnx", options);
    check_expand(p, 3, {3, 4, 4});
    check_expand(p, 2, {2, 4, 4});
}

TEST_CASE(expand_symbolic_input_ones_dims_test)
{
    migraphx::onnx_options options;
    options.use_symbolic_shapes   = true;
    options.default_dyn_dim_value = {1, 3};
    auto p                        = read_onnx("expand_dyn_input_ones_dims_test.onnx", options);
    EXPECT(p.get_output_shapes().back().symbolic());
    check_expand(p, 3, {3, 4, 4});
    check_expand(p, 2, {2, 4, 4});
}
