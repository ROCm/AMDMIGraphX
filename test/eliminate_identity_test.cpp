/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(*p.get_main_module(), {migraphx::eliminate_identity{}});
}

TEST_CASE(simple_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();

    auto one          = mm->add_literal(1);
    auto one_identity = mm->add_instruction(migraphx::make_op("identity"), one);
    auto two          = mm->add_literal(2);
    auto two_identity = mm->add_instruction(migraphx::make_op("identity"), two);
    mm->add_instruction(sum_op{}, one_identity, two_identity);
    run_pass(p);
    EXPECT(std::none_of(mm->begin(), mm->end(), [](const migraphx::instruction& ins) {
        return ins.name() == "identity";
    }));
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{3});
}

TEST_CASE(simple_test_end)
{
    migraphx::program p;

    auto* mm = p.get_main_module();

    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto ans = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(migraphx::make_op("identity"), ans);
    run_pass(p);
    EXPECT(std::none_of(mm->begin(), mm->end(), [](const migraphx::instruction& ins) {
        return ins.name() == "identity";
    }));
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{3});
}

TEST_CASE(simple_test_end_dependency)
{
    migraphx::program p;

    auto* mm = p.get_main_module();

    auto one   = mm->add_literal(1.0);
    auto two   = mm->add_literal(2.0);
    auto three = mm->add_literal(3.0);
    auto ans   = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(sum_op{}, ans, three);
    mm->add_instruction(migraphx::make_op("identity"), ans);
    run_pass(p);
    EXPECT(std::any_of(mm->begin(), mm->end(), [](const migraphx::instruction& ins) {
        return ins.name() == "identity";
    }));
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{3.0});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
