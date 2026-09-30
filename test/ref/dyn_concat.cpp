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
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>

#include <test.hpp>

TEST_CASE(dyn_concat_middle_axis)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto x       = mm->add_literal(migraphx::literal{{migraphx::shape::int32_type, {2, 3, 2}},
                                                     {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}});
    auto y       = mm->add_literal(migraphx::literal{{migraphx::shape::int32_type, {2, 2, 2}},
                                                     {100, 101, 102, 103, 104, 105, 106, 107}});
    auto x_count = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {2}});
    auto y_count = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {1}});
    auto result =
        mm->add_instruction(migraphx::make_op("dyn_concat", {{"axis", 1}}), x, y, x_count, y_count);
    auto buffer = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), result);
    auto count  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), result);
    mm->add_return({buffer, count});

    p.compile(migraphx::make_target("ref"));
    auto outputs = p.eval({});
    EXPECT(outputs.at(0).to_vector<int32_t>() == std::vector<int32_t>{0,   1,   2, 3, 100, 101, 0,
                                                                      0,   0,   0, 6, 7,   8,   9,
                                                                      104, 105, 0, 0, 0,   0});
    EXPECT(outputs.at(1).to_vector<int64_t>() == std::vector<int64_t>{3});
}

TEST_CASE(dyn_concat_zero_counts)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x =
        mm->add_literal(migraphx::literal{{migraphx::shape::float_type, {2, 2}}, {1, 2, 3, 4}});
    auto y    = mm->add_literal(migraphx::literal{{migraphx::shape::float_type, {1, 2}}, {5, 6}});
    auto zero = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    auto result =
        mm->add_instruction(migraphx::make_op("dyn_concat", {{"axis", 0}}), x, y, zero, zero);
    auto buffer = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), result);
    auto count  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), result);
    mm->add_return({buffer, count});

    p.compile(migraphx::make_target("ref"));
    auto outputs = p.eval({});
    EXPECT(outputs.at(0).to_vector<float>() == std::vector<float>(6, 0.0f));
    EXPECT(outputs.at(1).to_vector<int64_t>() == std::vector<int64_t>{0});
}
