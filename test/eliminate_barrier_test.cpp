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
#include <migraphx/eliminate_barrier.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>

#include <test.hpp>

TEST_CASE(remove_all_barriers)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {4}});
        auto a = m1.add_instruction(migraphx::make_op("barrier", {{"tag", "foo"}}), x);
        auto b = m1.add_instruction(migraphx::make_op("barrier", {{"tag", "bar"}}), a);
        auto r = m1.add_instruction(migraphx::make_op("relu"), b);
        m1.add_return({r});
    }
    migraphx::run_passes(m1, {migraphx::eliminate_barrier{}});

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {4}});
        auto r = m2.add_instruction(migraphx::make_op("relu"), x);
        m2.add_return({r});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(remove_tagged_barrier)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {4}});
        auto a = m1.add_instruction(migraphx::make_op("barrier", {{"tag", "foo"}}), x);
        auto b = m1.add_instruction(migraphx::make_op("barrier", {{"tag", "bar"}}), a);
        auto r = m1.add_instruction(migraphx::make_op("relu"), b);
        m1.add_return({r});
    }
    // Only the "foo" barrier is removed; the "bar" barrier is left in place.
    migraphx::run_passes(m1, {migraphx::eliminate_barrier{.tag = "foo"}});

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {4}});
        auto b = m2.add_instruction(migraphx::make_op("barrier", {{"tag", "bar"}}), x);
        auto r = m2.add_instruction(migraphx::make_op("relu"), b);
        m2.add_return({r});
    }
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
