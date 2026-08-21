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

TEST_CASE(symbolic_shape_arithmetic)
{
    migraphx::onnx_options options;
    options.use_symbolic_shapes     = true;
    options.map_dyn_input_dims["x"] = {{1, 1}, {1, 4}};
    auto p                          = read_onnx("symbolic_shape_arithmetic_test.onnx", options);

    const auto& input_dims = p.get_parameter_shapes().at("x").dyn_dims();
    const auto sequence    = input_dims[1].sym_expr;
    const auto outputs     = p.get_output_shapes();
    EXPECT(outputs.size() == 4);
    EXPECT(outputs[0] == migraphx::shape{migraphx::shape::float_type,
                                         {migraphx::shape::dynamic_dimension{
                                             sequence + migraphx::sym::lit(int64_t{1})}}});
    EXPECT(outputs[1] == migraphx::shape{migraphx::shape::float_type, {input_dims[1]}});
    EXPECT(outputs[2] ==
           migraphx::shape{
               migraphx::shape::float_type,
               {migraphx::shape::dynamic_dimension{migraphx::sym::lit(int64_t{2})},
                migraphx::shape::dynamic_dimension{sequence + migraphx::sym::lit(int64_t{1})},
                migraphx::shape::dynamic_dimension{sequence + migraphx::sym::lit(int64_t{1})},
                migraphx::shape::dynamic_dimension{sequence + sequence}}});
    EXPECT(outputs[3].dynamic());
    EXPECT(not outputs[3].symbolic());

    p.compile(migraphx::make_target("ref"));
    migraphx::shape input_shape{migraphx::shape::float_type, {1, 2}};
    std::vector<float> input(input_shape.elements(), 1.0f);
    auto results = p.eval({{"x", migraphx::argument{input_shape, input.data()}}});
    EXPECT(results[0].get_shape().lens() == std::vector<std::size_t>{3});
    EXPECT(results[1].get_shape().lens() == std::vector<std::size_t>{2});
    EXPECT(results[2].get_shape().lens() == std::vector<std::size_t>{2, 3, 3, 4});
    EXPECT(results[3].get_shape().lens() == std::vector<std::size_t>{2});
}
