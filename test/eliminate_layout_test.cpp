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
#include <migraphx/eliminate_layout.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::eliminate_layout{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(same_layout)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 2, 3, 4}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s);
        auto layout =
            m1.add_instruction(migraphx::make_op("layout", {{"permutation", {0, 1, 2, 3}}}), x);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), layout);
        m1.add_return({relu});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), x);
        m2.add_return({relu});
    }
    EXPECT(m1 == m2);
}

// The second layout gets a tensor that the first has already put in its layout
TEST_CASE(same_layout_not_standard)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 2, 3, 4}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s);
        auto layout1 =
            m1.add_instruction(migraphx::make_op("layout", {{"permutation", {0, 2, 3, 1}}}), x);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), layout1);
        auto layout2 =
            m1.add_instruction(migraphx::make_op("layout", {{"permutation", {0, 2, 3, 1}}}), relu);
        auto neg = m1.add_instruction(migraphx::make_op("neg"), layout2);
        m1.add_return({neg});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", s);
        auto layout =
            m2.add_instruction(migraphx::make_op("layout", {{"permutation", {0, 2, 3, 1}}}), x);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), layout);
        auto neg  = m2.add_instruction(migraphx::make_op("neg"), relu);
        m2.add_return({neg});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(different_layout)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 2, 3, 4}});
        auto layout =
            m1.add_instruction(migraphx::make_op("layout", {{"permutation", {0, 2, 3, 1}}}), x);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), layout);
        m1.add_return({relu});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// A transposed tensor has the lens of the layout's output but not its strides
TEST_CASE(transposed_input)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 3, 3, 4}});
        auto t =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), x);
        auto layout =
            m1.add_instruction(migraphx::make_op("layout", {{"permutation", {0, 1, 2, 3}}}), t);
        m1.add_return({layout});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// Only the instructions the target's matcher takes are removed
TEST_CASE(target_layout_matcher)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 2, 3, 4}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s);
        auto layout =
            m1.add_instruction(migraphx::make_op("layout", {{"permutation", {0, 1, 2, 3}}}), x);
        auto target_layout = m1.add_instruction(migraphx::make_op("identity"), layout);
        auto relu          = m1.add_instruction(migraphx::make_op("relu"), target_layout);
        m1.add_return({relu});
    }
    migraphx::run_passes(m1,
                         {migraphx::eliminate_layout{migraphx::match::name("identity")},
                          migraphx::dead_code_elimination{}});

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", s);
        auto layout =
            m2.add_instruction(migraphx::make_op("layout", {{"permutation", {0, 1, 2, 3}}}), x);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), layout);
        m2.add_return({relu});
    }
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
