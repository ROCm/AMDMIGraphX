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
#include <migraphx/rewrite_broadcast.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/module.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>

#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::rewrite_broadcast{}, migraphx::dead_code_elimination{}});
}

// Test that broadcast -> convert is rewritten to convert -> broadcast (single user)
TEST_CASE(broadcast_convert_single_user)
{
    migraphx::module m1;
    {
        // Input: x -> multibroadcast -> convert
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 2, 64, 1}});
        auto bcast = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 64, 32}}}), x);
        auto conv = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), bcast);
        m1.add_return({conv});
    }

    migraphx::module m2;
    {
        // Expected: x -> convert -> multibroadcast
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {1, 2, 64, 1}});
        auto conv = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), x);
        auto bcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 64, 32}}}), conv);
        m2.add_return({bcast});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

// Test that broadcast -> convert is NOT rewritten when broadcast has multiple users
TEST_CASE(broadcast_convert_multi_user_no_change)
{
    migraphx::module m1;
    {
        // Input: x -> multibroadcast -> [convert, add]
        // The multibroadcast has 2 users, so optimization should NOT fire
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 2, 64, 1}});
        auto y = m1.add_parameter("y", {migraphx::shape::float_type, {1, 2, 64, 32}});
        auto bcast = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 64, 32}}}), x);
        auto conv = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), bcast);
        auto add = m1.add_instruction(migraphx::make_op("add"), bcast, y);
        m1.add_return({conv, add});
    }

    migraphx::module m2 = m1; // Expected: no change

    run_pass(m1);
    EXPECT(m1 == m2);
}

// Test that broadcast -> reduce is rewritten to reduce -> broadcast (disjoint axes)
TEST_CASE(broadcast_reduce_disjoint_axes)
{
    migraphx::module m1;
    {
        // Input: x -> multibroadcast (axis 3) -> reduce_sum (axis 1)
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 2, 64, 1}});
        auto bcast = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 64, 32}}}), x);
        auto reduce = m1.add_instruction(
            migraphx::make_op("reduce_sum", {{"axes", {1}}}), bcast);
        m1.add_return({reduce});
    }

    migraphx::module m2;
    {
        // Expected: x -> reduce_sum (axis 1) -> multibroadcast
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {1, 2, 64, 1}});
        auto reduce = m2.add_instruction(
            migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto bcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 64, 32}}}), reduce);
        m2.add_return({bcast});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

// Test that broadcast -> reduce is NOT rewritten when axes overlap
TEST_CASE(broadcast_reduce_overlapping_axes_no_change)
{
    migraphx::module m1;
    {
        // Input: x -> multibroadcast (axis 3) -> reduce_sum (axis 3)
        // Axes overlap, so optimization should NOT fire
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 2, 64, 1}});
        auto bcast = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 64, 32}}}), x);
        auto reduce = m1.add_instruction(
            migraphx::make_op("reduce_sum", {{"axes", {3}}}), bcast);
        m1.add_return({reduce});
    }

    migraphx::module m2 = m1; // Expected: no change

    run_pass(m1);
    EXPECT(m1 == m2);
}

// Test combined: broadcast -> convert -> reduce chain gets fully optimized
TEST_CASE(broadcast_convert_reduce_chain)
{
    migraphx::module m1;
    {
        // Input: x -> multibroadcast -> convert -> reduce_sum
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 2, 64, 1}});
        auto bcast = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 64, 32}}}), x);
        auto conv = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), bcast);
        auto reduce = m1.add_instruction(
            migraphx::make_op("reduce_sum", {{"axes", {1}}}), conv);
        m1.add_return({reduce});
    }

    migraphx::module m2;
    {
        // Expected: x -> convert -> reduce_sum -> multibroadcast
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {1, 2, 64, 1}});
        auto conv = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), x);
        auto reduce = m2.add_instruction(
            migraphx::make_op("reduce_sum", {{"axes", {1}}}), conv);
        auto bcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 64, 32}}}), reduce);
        m2.add_return({bcast});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }

