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

// Tile x of shape {n, 1, 2} by {2, 3, 2}: each row {a, b} becomes 3 rows of {a, b, a, b},
// and the whole result is repeated twice along the first axis
static void check_tile(migraphx::program p, std::size_t n)
{
    p.compile(migraphx::make_target("ref"));
    std::vector<float> x(n * 2);
    std::iota(x.begin(), x.end(), 1);
    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument({migraphx::shape::float_type, {n, 1, 2}}, x.data());
    auto result = p.eval(pp).back();
    EXPECT(result.get_shape().lens() == std::vector<std::size_t>{2 * n, 3, 4});
    std::vector<float> gold;
    for(int t = 0; t < 2; t++)
    {
        for(std::size_t i = 0; i < n; i++)
        {
            for(int r = 0; r < 3; r++)
                gold.insert(gold.end(), {x[2 * i], x[2 * i + 1], x[2 * i], x[2 * i + 1]});
        }
    }
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    EXPECT(result_vector == gold);
}

TEST_CASE(tile_symbolic_input_test)
{
    migraphx::onnx_options options;
    options.use_symbolic_shapes   = true;
    options.default_dyn_dim_value = {1, 4};
    auto p                        = read_onnx("tile_dyn_input_test.onnx", options);
    EXPECT(p.get_output_shapes().back().symbolic());
    check_tile(p, 1);
    check_tile(p, 3);
}

TEST_CASE(tile_dyn_input_test)
{
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    auto p                        = read_onnx("tile_dyn_input_test.onnx", options);
    EXPECT(p.get_output_shapes().back().dynamic());
    check_tile(p, 1);
    check_tile(p, 3);
}
