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

TEST_CASE(symbolic_reshape_markers)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;

    const auto batch = var("batch", {1, 4});
    const auto input_shape =
        migraphx::shape{migraphx::shape::float_type, sym_dims({batch, lit(int64_t{4})})};

    migraphx::program expected;
    auto* mm = expected.get_main_module();
    auto x   = mm->add_parameter("x", input_shape);
    auto neg_one =
        mm->insert_literal(x, migraphx::literal{{migraphx::shape::int64_type, {1}}, {-1}});
    auto zero = mm->insert_literal(x, migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    mm->insert_literal(x, migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    auto index_1 =
        mm->insert_literal(x, migraphx::literal{migraphx::shape{migraphx::shape::int64_type}, {1}});
    auto index_0 =
        mm->insert_literal(x, migraphx::literal{migraphx::shape{migraphx::shape::int64_type}, {0}});
    auto x_shape =
        mm->add_instruction(migraphx::make_op("dimensions_of", {{"start", 0}, {"end", 2}}), x);
    auto batch_value =
        mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), x_shape, index_0);
    auto width_value =
        mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), x_shape, index_1);
    auto batch_vector =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), batch_value);
    auto width_vector =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), width_value);
    mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), zero, width_vector);
    mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), batch_vector, neg_one);

    const auto expressions = migraphx::to_value(std::vector<migraphx::sym::expr>{batch, lit(4)});
    auto zero_dims         = mm->add_instruction(
        migraphx::make_op("eval_expr_from_shape", {{"expressions", expressions}}), x);
    auto zero_allocation = mm->add_instruction(
        migraphx::make_op("allocate", {{"shape", migraphx::to_value(input_shape)}}), zero_dims);
    auto zero_output = mm->add_instruction(migraphx::make_op("reshape"), x, zero_allocation);

    auto inferred_dims = mm->add_instruction(
        migraphx::make_op("eval_expr_from_shape", {{"expressions", expressions}}), x);
    auto inferred_allocation = mm->add_instruction(
        migraphx::make_op("allocate", {{"shape", migraphx::to_value(input_shape)}}), inferred_dims);
    auto inferred_output =
        mm->add_instruction(migraphx::make_op("reshape"), x, inferred_allocation);
    mm->add_return({zero_output, inferred_output});

    migraphx::onnx_options options;
    options.use_symbolic_shapes     = true;
    options.map_dyn_input_dims["x"] = {{1, 4}, {4, 4}};
    auto p                          = read_onnx("symbolic_reshape_markers_test.onnx", options);

    EXPECT(expected == p);
}
