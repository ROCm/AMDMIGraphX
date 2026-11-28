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

TEST_CASE(pad_symbolic_pads_axis1_test)
{
    // The N columns of x are padded at the end with N zeros
    migraphx::onnx_options options;
    options.map_dyn_input_dims["x"] = {{2, 2}, {1, 8}};
    options.use_symbolic_shapes     = true;
    migraphx::program p             = read_onnx("pad_symbolic_pads_axis1_test.onnx", options);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape input_shape{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data = {1, 2, 3, 4, 5, 6};

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(input_shape, data.data());

    auto result = p.eval(pp).back();
    EXPECT(result.get_shape().lens() == std::vector<std::size_t>{2, 6});
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1, 2, 3, 0, 0, 0, 4, 5, 6, 0, 0, 0};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
