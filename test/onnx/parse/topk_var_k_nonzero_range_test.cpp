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

#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/sym.hpp>
#include <onnx_test.hpp>

TEST_CASE(topk_var_k_nonzero_range_test)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;
    auto count = var("main_NonZero_3", {0, 8});

    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto data = mm->add_parameter("data", {migraphx::shape::float_type, {{2, 2}, {1, 8}}});
    auto mask = mm->add_parameter("mask", {migraphx::shape::bool_type, {8}});
    auto count_index =
        mm->insert_literal(mm->end(), migraphx::literal{{migraphx::shape::int64_type, {1}}, {1}});
    auto nonzero = mm->add_instruction(migraphx::make_op("nonzero"), mask);
    auto indices =
        mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nonzero);
    auto nonzero_count =
        mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nonzero);
    auto nonzero_start =
        mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    indices = mm->add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {1}},
                           {"starts", {0}},
                           {"ends", migraphx::value::array{migraphx::to_value(count)}}}),
        indices,
        nonzero_start,
        nonzero_count);
    auto nonzero_shape = mm->add_instruction(
        migraphx::make_op("dimensions_of", {{"start", 0}, {"end", 2}}), indices);
    auto k =
        mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), nonzero_shape, count_index);

    auto topk = mm->add_instruction(
        migraphx::make_op("topk", {{"k", 8}, {"axis", 1}, {"largest", true}}), data);
    auto values = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), topk);
    auto topk_indices =
        mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), topk);
    std::vector<migraphx::shape::dynamic_dimension> output_dims = {
        migraphx::shape::dynamic_dimension{lit(2)}, migraphx::shape::dynamic_dimension{count}};
    migraphx::instruction::replace(values,
                                   values->get_operator(),
                                   migraphx::shape{migraphx::shape::float_type, output_dims},
                                   values->inputs());
    migraphx::instruction::replace(topk_indices,
                                   topk_indices->get_operator(),
                                   migraphx::shape{migraphx::shape::int64_type, output_dims},
                                   topk_indices->inputs());

    auto topk_start = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    auto slice      = migraphx::make_op("dyn_slice",
                                        {{"axes", {1}},
                                         {"starts", {0}},
                                         {"ends", migraphx::value::array{migraphx::to_value(count)}}});
    values          = mm->add_instruction(slice, values, topk_start, k);
    topk_indices    = mm->add_instruction(slice, topk_indices, topk_start, k);
    mm->add_return({values, topk_indices});

    migraphx::onnx_options options;
    options.map_dyn_input_dims["data"] = {{2, 2}, {1, 8}};
    auto prog                          = read_onnx("topk_var_k_nonzero_range_test.onnx", options);
    EXPECT(p == prog);
}
