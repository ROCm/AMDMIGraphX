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

TEST_CASE(symbolic_reshape_data_root_test)
{
    using dd = migraphx::shape::dynamic_dimension;
    using migraphx::sym::lit;
    using migraphx::sym::var;

    const auto count        = var("main_NonZero_1", {0, 4});
    const auto sliced_count = migraphx::sym::min(count, lit(4));
    migraphx::program p;
    auto* mm       = p.get_main_module();
    auto condition = mm->add_parameter("condition", {migraphx::shape::bool_type, {4}});
    auto nonzero   = mm->add_instruction(migraphx::make_op("nonzero"), condition);
    auto indices =
        mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nonzero);
    auto num_nonzero =
        mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nonzero);
    auto starts = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    auto sliced = mm->add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {1}},
                           {"starts", {0}},
                           {"ends", migraphx::value::array{migraphx::to_value(count)}}}),
        indices,
        starts,
        num_nonzero);
    auto transposed =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), sliced);
    auto transposed_shape = mm->add_instruction(
        migraphx::make_op("dimensions_of", {{"start", 0}, {"end", 2}}), transposed);
    auto trailing_dims =
        mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {2}}, {1, 1}});
    mm->add_instruction(
        migraphx::make_op("concat", {{"axis", 0}}), transposed_shape, trailing_dims);
    const auto expressions =
        migraphx::to_value(std::vector<migraphx::sym::expr>{sliced_count, lit(1), lit(1), lit(1)});
    auto target_dims = mm->add_instruction(
        migraphx::make_op("eval_expr_from_shape", {{"expressions", expressions}}), transposed);
    const migraphx::shape output_shape{migraphx::shape::int64_type,
                                       {dd{sliced_count}, dd{lit(1)}, dd{lit(1)}, dd{lit(1)}}};
    auto allocation = mm->add_instruction(
        migraphx::make_op("allocate", {{"shape", migraphx::to_value(output_shape)}}), target_dims);
    mm->add_return({mm->add_instruction(migraphx::make_op("reshape"), transposed, allocation)});

    EXPECT(p == read_onnx("symbolic_reshape_data_root_test.onnx"));
}
