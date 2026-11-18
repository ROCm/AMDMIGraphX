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
#include <numeric>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

static std::vector<float> run_pad_symbolic_pads_begin_value(std::size_t n)
{
    migraphx::onnx_options options;
    options.map_dyn_input_dims["x"] = {{2, 2}, {1, 8}};
    options.use_symbolic_shapes     = true;
    migraphx::program p             = read_onnx("pad_symbolic_pads_begin_value_test.onnx", options);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape input_shape{migraphx::shape::float_type, {2, n}};
    std::vector<float> data(2 * n);
    std::iota(data.begin(), data.end(), 1.0f);
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(input_shape, data.data());

    auto result = p.eval(pp).back();
    EXPECT(result.get_shape().lens() == std::vector<std::size_t>{2, 9});
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    return result_vector;
}

TEST_CASE(pad_symbolic_pads_begin_value_test)
{
    // One 5 before the N columns and 8 - N after them
    std::vector<float> gold3 = {5, 1, 2, 3, 5, 5, 5, 5, 5, 5, 4, 5, 6, 5, 5, 5, 5, 5};
    EXPECT(migraphx::verify::verify_rms_range(run_pad_symbolic_pads_begin_value(3), gold3));
    // The end pad is 0 at run time
    std::vector<float> gold8 = {5, 1, 2, 3, 4, 5, 6, 7, 8, 5, 9, 10, 11, 12, 13, 14, 15, 16};
    EXPECT(migraphx::verify::verify_rms_range(run_pad_symbolic_pads_begin_value(8), gold8));
}
