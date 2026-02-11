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
 *
 */
#include <migraphx/propagate_precision.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/common.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m,
                         {migraphx::propagate_precision{},
                          migraphx::eliminate_common_subexpression{},
                          migraphx::dead_code_elimination{}});
}

TEST_CASE(propagate_input)
{
    migraphx::shape s1{migraphx::shape::half_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 3}};
    migraphx::module m1;
    {
        auto x        = m1.add_parameter("x", s1);
        auto y        = m1.add_parameter("y", s2);
        auto two      = m1.add_literal(migraphx::literal{{migraphx::shape::half_type}, {2}});
        auto div      = migraphx::add_common_op(m1, migraphx::make_op("div"), {x, two});
        auto sqrt     = m1.add_instruction(migraphx::make_op("sqrt"), div);
        auto convert1 = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), sqrt);
        auto mul      = m1.add_instruction(migraphx::make_op("mul"), convert1, y);
        auto convert2 = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), mul);
        m1.add_return({convert2});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x        = m2.add_parameter("x", s1);
        auto y        = m2.add_parameter("y", s2);
        auto convert1 = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), x);
        auto two      = m2.add_literal(migraphx::literal{{migraphx::shape::half_type}, {2}});
        auto div      = migraphx::add_common_op(m2, migraphx::make_op("div"), {convert1, two});
        auto sqrt     = m2.add_instruction(migraphx::make_op("sqrt"), div);
        auto mul      = m2.add_instruction(migraphx::make_op("mul"), sqrt, y);
        auto convert2 = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), mul);
        m2.add_return({convert2});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(propagate_output)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::half_type, {2, 3}};
    migraphx::module m1;
    {
        auto x        = m1.add_parameter("x", s1);
        auto y        = m1.add_parameter("y", s2);
        auto convert1 = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), x);
        auto two  = m1.add_literal(migraphx::literal{{migraphx::shape::half_type}, {2}});
        auto div  = migraphx::add_common_op(m1, migraphx::make_op("div"), {convert1, two});
        auto sqrt = m1.add_instruction(migraphx::make_op("sqrt"), div);
        auto mul  = m1.add_instruction(migraphx::make_op("mul"), sqrt, y);
        m1.add_return({mul});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x        = m2.add_parameter("x", s1);
        auto y        = m2.add_parameter("y", s2);
        auto two      = m2.add_literal(migraphx::literal{{migraphx::shape::half_type}, {2}});
        auto div      = migraphx::add_common_op(m2, migraphx::make_op("div"), {x, two});
        auto sqrt     = m2.add_instruction(migraphx::make_op("sqrt"), div);
        auto convert1 = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), sqrt);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), convert1, y);
        m2.add_return({mul});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(propagate_conflict)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::double_type, {2, 3}};
    migraphx::module m1;
    {
        auto x        = m1.add_parameter("x", s1);
        auto y        = m1.add_parameter("y", s2);
        auto convert1 = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), x);
        auto two      = m1.add_literal(migraphx::literal{{migraphx::shape::half_type}, {2}});
        auto div      = migraphx::add_common_op(m1, migraphx::make_op("div"), {convert1, two});
        auto sqrt     = m1.add_instruction(migraphx::make_op("sqrt"), div);
        auto convert2 = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::double_type}}), sqrt);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), convert2, y);
        m1.add_return({mul});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x        = m2.add_parameter("x", s1);
        auto y        = m2.add_parameter("y", s2);
        auto convert1 = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::double_type}}), x);
        auto two  = m2.add_literal(migraphx::literal{{migraphx::shape::half_type}, {2}});
        auto div  = migraphx::add_common_op(m2, migraphx::make_op("div"), {convert1, two});
        auto sqrt = m2.add_instruction(migraphx::make_op("sqrt"), div);
        auto mul  = m2.add_instruction(migraphx::make_op("mul"), sqrt, y);
        m2.add_return({mul});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(propagate_reduce)
{
    migraphx::shape s1{migraphx::shape::half_type, {2, 3}};
    migraphx::module m1;
    {
        auto x        = m1.add_parameter("x", s1);
        auto three    = m1.add_literal(migraphx::literal{{migraphx::shape::half_type}, {3}});
        auto squared  = m1.add_instruction(migraphx::make_op("mul"), x, x);
        auto div      = migraphx::add_common_op(m1, migraphx::make_op("div"), {squared, three});
        auto convert1 = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), div);
        auto reduce =
            m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), convert1);
        auto convert2 = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), reduce);
        auto sqrt = m1.add_instruction(migraphx::make_op("sqrt"), convert2);
        auto mul  = migraphx::add_common_op(m1, migraphx::make_op("mul"), {x, sqrt});
        m1.add_return({mul});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x        = m2.add_parameter("x", s1);
        auto convert1 = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), x);
        auto three    = m2.add_literal(migraphx::literal{{migraphx::shape::half_type}, {3}});
        auto squared  = m2.add_instruction(migraphx::make_op("mul"), convert1, convert1);
        auto div      = migraphx::add_common_op(m2, migraphx::make_op("div"), {squared, three});
        auto reduce   = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), div);
        auto sqrt     = m2.add_instruction(migraphx::make_op("sqrt"), reduce);
        auto convert2 = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), sqrt);
        auto mul = migraphx::add_common_op(m2, migraphx::make_op("mul"), {x, convert2});
        m2.add_return({mul});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(propagate_reduce_float_to_double)
{
    migraphx::shape s1{migraphx::shape::double_type, {3, 4}, {4, 1}};
    migraphx::module m1;
    {
        auto x        = m1.add_parameter("x", s1);
        auto convert1 = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), x);
        auto reduce =
            m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), convert1);
        auto convert2 = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::double_type}}), reduce);
        auto squeeze = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), convert2);
        m1.add_return({squeeze});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x       = m2.add_parameter("x", s1);
        auto reduce  = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto squeeze = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), reduce);
        m2.add_return({squeeze});
    }
    EXPECT(m1.sort() == m2.sort());
}

// Verify that propagation stops at type category boundaries (int div -> float)
TEST_CASE(propagate_type_category_boundary)
{
    migraphx::shape s1{migraphx::shape::int32_type, {1, 4}};
    migraphx::module m1;
    {
        auto x        = m1.add_parameter("x", s1);
        auto two      = m1.add_literal(migraphx::literal{{migraphx::shape::int32_type}, {2}});
        auto int_div  = migraphx::add_common_op(m1, migraphx::make_op("div"), {x, two});
        auto convert1 = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), int_div);
        auto one       = m1.add_literal(migraphx::literal{{migraphx::shape::float_type}, {1.0f}});
        auto float_add = migraphx::add_common_op(m1, migraphx::make_op("add"), {convert1, one});
        auto convert2  = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::double_type}}),
            float_add);
        m1.add_return({convert2});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x        = m2.add_parameter("x", s1);
        auto two      = m2.add_literal(migraphx::literal{{migraphx::shape::int32_type}, {2}});
        auto int_div  = migraphx::add_common_op(m2, migraphx::make_op("div"), {x, two});
        auto convert1 = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::double_type}}), int_div);
        auto one       = m2.add_literal(migraphx::literal{{migraphx::shape::float_type}, {1.0f}});
        auto float_add = migraphx::add_common_op(m2, migraphx::make_op("add"), {convert1, one});
        m2.add_return({float_add});
    }
    EXPECT(m1.sort() == m2.sort());
}

// Verify that int precision propagation through a bool/int op chain
// does not cross back into upstream float pointwise ops
TEST_CASE(propagate_no_crossover_through_bool_int_chain)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 32}};
    migraphx::module m1;
    {
        auto x        = m1.add_parameter("x", s1);
        auto sig      = m1.add_instruction(migraphx::make_op("sigmoid"), x);
        auto cvt_bool = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), sig);
        auto n        = m1.add_instruction(migraphx::make_op("not"), cvt_bool);
        auto cvt_int  = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int64_type}}), n);
        auto rsum =
            m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), cvt_int);
        m1.add_return({rsum});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x       = m2.add_parameter("x", s1);
        auto sig     = m2.add_instruction(migraphx::make_op("sigmoid"), x);
        auto cvt_int = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int64_type}}), sig);
        auto n    = m2.add_instruction(migraphx::make_op("not"), cvt_int);
        auto rsum = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), n);
        m2.add_return({rsum});
    }
    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
