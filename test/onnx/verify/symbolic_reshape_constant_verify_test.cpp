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
#include <onnx_test.hpp>

TEST_CASE(symbolic_reshape_constant_verify_test)
{
    migraphx::onnx_options options;
    options.use_symbolic_shapes     = true;
    options.map_dyn_input_dims["x"] = {{1, 4}, {4, 4}};
    auto p                          = read_onnx("symbolic_reshape_constant_test.onnx", options);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape input_shape{migraphx::shape::float_type, {3, 4}};
    std::vector<float> input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    migraphx::shape other_shape{migraphx::shape::float_type, {2}};
    std::vector<float> other = {12, 13};
    auto results             = p.eval({{"other", migraphx::argument{other_shape, other.data()}},
                                       {"x", migraphx::argument{input_shape, input.data()}}});

    EXPECT(results.size() == 2);
    EXPECT(results.at(0).get_shape() == input_shape);
    EXPECT(results.at(1).get_shape() == input_shape);
    std::vector<float> result_data;
    results.at(0).visit([&](auto output) { result_data.assign(output.begin(), output.end()); });
    EXPECT(result_data == input);
    results.at(1).visit([&](auto output) { result_data.assign(output.begin(), output.end()); });
    EXPECT(result_data == input);
}
