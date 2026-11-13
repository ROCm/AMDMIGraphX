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

TEST_CASE(symbolic_reshape_if_test)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;

    const auto batch = var("batch", {1, 4});
    const auto input_shape =
        migraphx::shape{migraphx::shape::float_type, sym_dims({batch, lit(int64_t{4})})};

    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto cond = mm->add_parameter("cond", {migraphx::shape::bool_type});
    auto x    = mm->add_parameter("x", input_shape);

    auto* then_mod = p.create_module("main_If_2_if");
    const auto expressions = migraphx::to_value(std::vector<migraphx::sym::expr>{batch, lit(4)});
    auto then_dims         = then_mod->add_instruction(
        migraphx::make_op("eval_expr_from_shape", {{"expressions", expressions}}), x);
    auto then_allocation = then_mod->add_instruction(
        migraphx::make_op("allocate", {{"shape", migraphx::to_value(input_shape)}}), then_dims);
    auto then_output = then_mod->add_instruction(migraphx::make_op("reshape"), x, then_allocation);
    then_mod->add_return({then_output});

    auto* else_mod = p.create_module("main_If_2_else");
    auto else_dims = else_mod->add_instruction(
        migraphx::make_op("eval_expr_from_shape", {{"expressions", expressions}}), x);
    auto else_allocation = else_mod->add_instruction(
        migraphx::make_op("allocate", {{"shape", migraphx::to_value(input_shape)}}), else_dims);
    auto else_output = else_mod->add_instruction(migraphx::make_op("reshape"), x, else_allocation);
    else_mod->add_return({else_output});

    auto result = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
    auto output = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), result);
    mm->add_return({output});

    migraphx::onnx_options options;
    options.map_dyn_input_dims["x"] = input_shape.dyn_dims();
    options.use_symbolic_shapes     = true;
    EXPECT(p == read_onnx("symbolic_reshape_if_test.onnx", options));
}
