/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/optimize_module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/serialize.hpp>
#include <test.hpp>

void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::optimize_module{}});
}

TEST_CASE(broadcast_transpose_inner_broadcast)
{
    // first optimizes broadcast+transpose to just a broadcast,
    // then finds inner broadcast to become mul+broadcast
    migraphx::module m1;
    {
        auto l1 = m1.add_parameter("x", {migraphx::shape::float_type, {1}, {0}});
        auto l2 = m1.add_parameter("y", {migraphx::shape::float_type, {1}, {0}});
        auto mb1 =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3}}}), l1);
        auto mb2 = m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 2}}}), l2);
        auto t1 = m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), mb1);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), mb2, t1);
        m1.add_return({mul});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto l1 = m2.add_parameter("x", {migraphx::shape::float_type, {1}, {0}});
        auto l2 = m2.add_parameter("y", {migraphx::shape::float_type, {1}, {0}});
        auto mul = m2.add_instruction(migraphx::make_op("mul"), l2, l1);
        auto mb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 2}}}), mul);
        m2.add_return({mb});
    }
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
