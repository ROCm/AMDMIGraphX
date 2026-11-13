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

#include <onnx_test.hpp>

TEST_CASE(symbolic_reshape_loop_test)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;

    const auto batch = var("batch", {1, 4});
    const auto input_shape =
        migraphx::shape{migraphx::shape::float_type, sym_dims({batch, lit(int64_t{4})})};
    const migraphx::shape iteration_shape{migraphx::shape::int64_type, {1}};
    const migraphx::shape condition_shape{migraphx::shape::bool_type, {1}};
    const migraphx::shape carried_shape{migraphx::shape::float_type, {1, 1}};

    migraphx::program p;
    auto* mm           = p.get_main_module();
    auto max_iteration = mm->add_parameter("max_trip_count", iteration_shape);
    auto condition     = mm->add_parameter("cond", condition_shape);
    auto carried       = mm->add_parameter("carried", carried_shape);
    auto x             = mm->add_parameter("x", input_shape);

    auto* body = p.create_module("main_Loop_4_loop");
    body->add_parameter("iteration_num", iteration_shape);
    auto body_condition = body->add_parameter("body_cond", condition_shape);
    body->add_parameter("body_carried", carried_shape);
    const auto expressions = migraphx::to_value(std::vector<migraphx::sym::expr>{batch, lit(4)});
    auto body_dims         = body->add_instruction(
        migraphx::make_op("eval_expr_from_shape", {{"expressions", expressions}}), x);
    auto body_allocation = body->add_instruction(
        migraphx::make_op("allocate", {{"shape", migraphx::to_value(input_shape)}}), body_dims);
    auto body_output = body->add_instruction(migraphx::make_op("reshape"), x, body_allocation);
    auto reduced_output =
        body->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 1}}}), body_output);
    body->add_return({body_condition, reduced_output});

    auto result = mm->add_instruction(migraphx::make_op("loop", {{"max_iterations", 10}}),
                                      {max_iteration, condition, carried},
                                      {body});
    auto carried_output =
        mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), result);
    mm->add_return({carried_output});

    migraphx::onnx_options options;
    options.map_dyn_input_dims["x"] = input_shape.dyn_dims();
    options.use_symbolic_shapes     = true;
    EXPECT(p == read_onnx("symbolic_reshape_loop_test.onnx", options));
}
