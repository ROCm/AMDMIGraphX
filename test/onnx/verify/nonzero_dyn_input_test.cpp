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

// Run a dynamic input with 3 of its 4 possible rows: the operator pads the indices for the 4x2
// maximum and the parser's trim still cuts them down to the elements that are actually nonzero.
TEST_CASE(nonzero_dyn_input_test)
{
    migraphx::onnx_options options;
    options.map_dyn_input_dims["data"] = {{1, 4}, {2, 2}};
    auto p                             = read_onnx("nonzero_dynamic_test.onnx", options);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::bool_type, {3, 2}};
    std::vector<char> data = {1, 0, 1, 1, 0, 1};

    migraphx::parameter_map pp;
    pp["data"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<int64_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // np.nonzero(data.reshape(3, 2)) is ((0, 1, 1, 2), (0, 0, 1, 1)).
    std::vector<int64_t> gold = {0, 1, 1, 2, 0, 0, 1, 1};
    EXPECT(result_vector == gold);
    // The trim is an aliased view into the buffer padded out to 8 columns, so it keeps that
    // buffer's row stride.
    EXPECT(result.get_shape() == migraphx::shape{migraphx::shape::int64_type, {2, 4}, {8, 1}});
}
