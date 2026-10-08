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
#include <migraphx/promote_storage_type.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m,
                         {migraphx::promote_storage_type{{migraphx::shape::bf16_type}},
                          migraphx::dead_code_elimination{}});
}

TEST_CASE(promote_pointwise)
{
    migraphx::shape s{migraphx::shape::bf16_type, {2, 3}};
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", s);
        auto y   = m1.add_parameter("y", s);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), x, y);
        m1.add_return({mul});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", s);
        auto y  = m2.add_parameter("y", s);
        auto xc = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), x);
        auto yc = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), y);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), xc, yc);
        auto cb  = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bf16_type}}), mul);
        m2.add_return({cb});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(promote_adjacent_pointwise_reduce)
{
    migraphx::shape s{migraphx::shape::bf16_type, {2, 8}};
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", s);
        auto y   = m1.add_parameter("y", s);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), x, y);
        auto rs  = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), mul);
        m1.add_return({rs});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", s);
        auto y  = m2.add_parameter("y", s);
        auto xc = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), x);
        auto yc = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), y);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), xc, yc);
        // No convert back and forth between the mul and the reduction
        auto rs = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), mul);
        auto cb = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bf16_type}}), rs);
        m2.add_return({cb});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(promote_reduce_only)
{
    migraphx::shape s{migraphx::shape::bf16_type, {2, 8}};
    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", s);
        auto rs = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        m1.add_return({rs});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", s);
        auto xc = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), x);
        auto rs = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), xc);
        auto cb = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bf16_type}}), rs);
        m2.add_return({cb});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(promote_where_keeps_condition)
{
    migraphx::shape s{migraphx::shape::bf16_type, {2, 3}};
    migraphx::shape bs{migraphx::shape::bool_type, {2, 3}};
    migraphx::module m1;
    {
        auto c = m1.add_parameter("c", bs);
        auto x = m1.add_parameter("x", s);
        auto y = m1.add_parameter("y", s);
        auto w = m1.add_instruction(migraphx::make_op("where"), c, x, y);
        m1.add_return({w});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto c  = m2.add_parameter("c", bs);
        auto x  = m2.add_parameter("x", s);
        auto y  = m2.add_parameter("y", s);
        auto xc = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), x);
        auto yc = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), y);
        auto w  = m2.add_instruction(migraphx::make_op("where"), c, xc, yc);
        auto cb = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bf16_type}}), w);
        m2.add_return({cb});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(promote_unlisted_type_unchanged)
{
    migraphx::shape s{migraphx::shape::half_type, {2, 3}};
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", s);
        auto y   = m1.add_parameter("y", s);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), x, y);
        m1.add_return({mul});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(promote_idempotent)
{
    migraphx::shape s{migraphx::shape::bf16_type, {2, 8}};
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", s);
        auto y   = m1.add_parameter("y", s);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), x, y);
        auto rs  = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), mul);
        m1.add_return({rs});
    }
    run_pass(m1);
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
