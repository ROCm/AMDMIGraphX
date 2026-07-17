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

static std::vector<float> run_onnx(const std::string& target)
{
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {3, 8};
    options.use_symbolic_shapes   = true;
    auto p                        = read_onnx("expand_dyn_input_static_dims_throw.onnx", options);
    p.compile(migraphx::make_target(target));

    migraphx::shape sx{migraphx::shape::float_type, {3, 1, 1}};
    std::vector<float> data(sx.elements());
    std::iota(data.begin(), data.end(), 1.0f);
    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument(sx, data.data());
    auto result = p.eval(pp).back();

    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    return result_vector;
}

TEST_CASE(expand_dyn_input_static_dims_test)
{
    std::vector<float> gold(48);
    for(std::size_t i = 0; i < 3; ++i)
    {
        const float v = static_cast<float>(i + 1);
        std::fill_n(gold.begin() + i * 16, 16, v);
    }

    auto ref_result = run_onnx("ref");

    // print gold, gpu_result, ref_result
    std::cout << "gold: " << std::endl;
    for(auto g : gold)
    {
        std::cout << g << " ";
    }
    std::cout << std::endl;
    std::cout << "ref_result: " << std::endl;
    for(auto g : ref_result)
    {
        std::cout << g << " ";
    }
    std::cout << std::endl;

    EXPECT(migraphx::verify::verify_rms_range(ref_result, gold));

    auto gpu_result = run_onnx("gpu");
    std::cout << "gpu_result: " << std::endl;
    for(auto g : gpu_result)
    {
        std::cout << g << " ";
    }
    std::cout << std::endl;
    EXPECT(migraphx::verify::verify_rms_range(gpu_result, gold));
    EXPECT(migraphx::verify::verify_rms_range(gpu_result, ref_result));
}
