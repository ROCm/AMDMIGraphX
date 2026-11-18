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

// x {2, 1, 2} is tiled by {Shape(y)[0], 3, 2}, where y is {1, k}: the first repeat is 1 for any k
static void check_tile(migraphx::program p, std::size_t k)
{
    p.compile(migraphx::make_target("ref"));
    std::vector<float> x = {1, 2, 3, 4};
    std::vector<float> y(k);
    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument({migraphx::shape::float_type, {2, 1, 2}}, x.data());
    pp["y"]     = migraphx::argument({migraphx::shape::float_type, {1, k}}, y.data());
    auto result = p.eval(pp).back();
    EXPECT(result.get_shape().lens() == std::vector<std::size_t>{2, 3, 4});
    std::vector<float> gold;
    for(std::size_t i = 0; i < 2; i++)
    {
        for(int r = 0; r < 3; r++)
            gold.insert(gold.end(), {x[2 * i], x[2 * i + 1], x[2 * i], x[2 * i + 1]});
    }
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    EXPECT(result_vector == gold);
}

TEST_CASE(tile_shape_repeats_symbolic_test)
{
    migraphx::onnx_options options;
    options.use_symbolic_shapes   = true;
    options.default_dyn_dim_value = {1, 4};
    auto p                        = read_onnx("tile_shape_repeats_test.onnx", options);
    check_tile(p, 1);
    check_tile(p, 3);
}

// Without symbolic shapes the repeats have no value at parse time
TEST_CASE(tile_shape_repeats_dyn_throw)
{
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    EXPECT(test::throws([&] { read_onnx("tile_shape_repeats_test.onnx", options); }));
}
