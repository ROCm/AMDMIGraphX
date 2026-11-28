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

#include <cmath>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

// x is {1, 3, 2}; scale and bias are 1-D over the last axis
static std::vector<float> eval_layer_norm(migraphx::program p)
{
    p.compile(migraphx::make_target("ref"));
    std::vector<float> x     = {1, 2, 3, 5, -1, 3};
    std::vector<float> scale = {2, 0.5};
    std::vector<float> bias  = {1, -1};
    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument({migraphx::shape::float_type, {1, 3, 2}}, x.data());
    pp["scale"] = migraphx::argument({migraphx::shape::float_type, {2}}, scale.data());
    pp["bias"]  = migraphx::argument({migraphx::shape::float_type, {2}}, bias.data());
    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    return result_vector;
}

static std::vector<float> gold()
{
    // Each pair {a, b} with a < b normalizes to {-d, d}, d = h / sqrt(h * h + eps), h = (b - a) / 2
    std::vector<float> result;
    for(float h : {0.5f, 1.0f, 2.0f})
    {
        float d = h / std::sqrt(h * h + 1e-5f);
        result.push_back(-d * 2 + 1);
        result.push_back(d * 0.5f - 1);
    }
    return result;
}

TEST_CASE(layer_norm_dyn_test)
{
    migraphx::onnx_options options;
    options.map_dyn_input_dims["x"] = {{1, 1}, {1, 4}, {2, 2}};
    auto p                          = read_onnx("layer_norm_3d_test.onnx", options);
    EXPECT(migraphx::verify::verify_rms_range(eval_layer_norm(p), gold()));
}

TEST_CASE(layer_norm_symbolic_test)
{
    migraphx::onnx_options options;
    options.use_symbolic_shapes     = true;
    options.map_dyn_input_dims["x"] = {{1, 1}, {1, 4}, {2, 2}};
    auto p                          = read_onnx("layer_norm_3d_test.onnx", options);
    EXPECT(p.get_output_shapes().back().symbolic());
    EXPECT(migraphx::verify::verify_rms_range(eval_layer_norm(p), gold()));
}
