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
#include <migraphx/verify.hpp>

#include <test.hpp>

// Only the strict lower triangle of A is read: its diagonal and upper half are
// filled with values that would change the result if they were used
TEST_CASE(triangular_solve_unit_lower_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape as{migraphx::shape::float_type, {3, 3}};
    migraphx::shape bs{migraphx::shape::float_type, {3, 2}};
    auto a = mm->add_literal(migraphx::literal{as, {7, 9, 9, 2, 7, 9, -1, 3, 7}});
    auto b = mm->add_literal(migraphx::literal{bs, {1, 2, 3, 4, 5, 6}});
    mm->add_instruction(migraphx::make_op("triangular_solve", {{"unit_diagonal", true}}), a, b);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back().to_vector<float>();
    std::vector<float> gold{1, 2, 1, 0, 3, 8};
    EXPECT(migraphx::verify::verify_rms_range(result, gold));
}

TEST_CASE(triangular_solve_lower_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape as{migraphx::shape::float_type, {3, 3}};
    migraphx::shape bs{migraphx::shape::float_type, {3, 2}};
    auto a = mm->add_literal(migraphx::literal{as, {2, 0, 0, 1, 4, 0, -2, 3, 8}});
    auto b = mm->add_literal(migraphx::literal{bs, {1, 2, 3, 4, 5, 6}});
    mm->add_instruction(migraphx::make_op("triangular_solve"), a, b);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back().to_vector<float>();
    std::vector<float> gold{0.5, 1, 0.625, 0.75, 0.515625, 0.71875};
    EXPECT(migraphx::verify::verify_rms_range(result, gold));
}

TEST_CASE(triangular_solve_upper_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape as{migraphx::shape::float_type, {3, 3}};
    migraphx::shape bs{migraphx::shape::float_type, {3, 2}};
    auto a = mm->add_literal(migraphx::literal{as, {2, 1, -1, 0, 4, 3, 0, 0, 8}});
    auto b = mm->add_literal(migraphx::literal{bs, {1, 2, 3, 4, 5, 6}});
    mm->add_instruction(migraphx::make_op("triangular_solve", {{"lower", false}}), a, b);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back().to_vector<float>();
    std::vector<float> gold{0.671875, 1.15625, 0.28125, 0.4375, 0.625, 0.75};
    EXPECT(migraphx::verify::verify_rms_range(result, gold));
}

// A broadcast over the batch axis is read through its strides
TEST_CASE(triangular_solve_broadcast_batch_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape as{migraphx::shape::float_type, {1, 3, 3}};
    migraphx::shape bs{migraphx::shape::float_type, {2, 3, 1}};
    auto a = mm->add_literal(migraphx::literal{as, {7, 9, 9, 2, 7, 9, -1, 3, 7}});
    a      = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 3}}}),
                            a);
    auto b = mm->add_literal(migraphx::literal{bs, std::vector<float>{1, 3, 5, 2, -1, 0.5}});
    mm->add_instruction(migraphx::make_op("triangular_solve", {{"unit_diagonal", true}}), a, b);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back().to_vector<float>();
    std::vector<float> gold{1, 1, 3, 2, -5, 17.5};
    EXPECT(migraphx::verify::verify_rms_range(result, gold));
}
