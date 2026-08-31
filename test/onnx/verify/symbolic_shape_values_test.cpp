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

static void eval_symbolic_shape_values(migraphx::program& p, std::size_t sequence)
{
    migraphx::shape input_shape{migraphx::shape::float_type, {1, sequence}};
    std::vector<float> input(input_shape.elements(), 1.0f);
    migraphx::parameter_map params;
    params["x"] = migraphx::argument{input_shape, input.data()};

    auto results = p.eval(params);
    EXPECT(results.size() == 2);
    EXPECT(results[0].get_shape().lens() == std::vector<std::size_t>{1, 1, sequence, sequence, 2});
    EXPECT(results[1].get_shape().lens() == std::vector<std::size_t>{1, 1, sequence, sequence});
}

TEST_CASE(symbolic_shape_values)
{
    migraphx::onnx_options options;
    options.use_symbolic_shapes     = true;
    options.map_dyn_input_dims["x"] = {{1, 1}, {1, 4}};
    auto p                          = read_onnx("symbolic_shape_values_test.onnx", options);

    const auto input = p.get_parameter_shapes().at("x");
    EXPECT(input.symbolic());
    const auto& input_dims = input.dyn_dims();
    std::vector<migraphx::shape::dynamic_dimension> output_dims{
        input_dims[0], input_dims[0], input_dims[1], input_dims[1]};
    auto sliced_dims = output_dims;
    sliced_dims.push_back(migraphx::shape::dynamic_dimension{migraphx::sym::lit(2)});
    const auto& outputs = p.get_output_shapes();
    EXPECT(outputs.size() == 2);
    EXPECT(outputs[0].type() == migraphx::shape::int64_type);
    EXPECT(outputs[0].dyn_dims() == sliced_dims);
    EXPECT(outputs[1] == migraphx::shape{migraphx::shape::float_type, output_dims});

    p.compile(migraphx::make_target("ref"));
    eval_symbolic_shape_values(p, 2);
    eval_symbolic_shape_values(p, 4);
}
