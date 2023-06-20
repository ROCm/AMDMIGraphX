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
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>

#include <test.hpp>

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::dead_code_elimination{}});
}

TEST_CASE(simple_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(sum_op{}, one, two);
    auto count = std::distance(mm->begin(), mm->end());
    run_pass(p);
    EXPECT(std::distance(mm->begin(), mm->end()) == count);
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(simple_test_nop)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(nop{});
    mm->add_instruction(sum_op{}, one, two);
    auto count = std::distance(mm->begin(), mm->end());
    run_pass(p);
    EXPECT(std::distance(mm->begin(), mm->end()) == count);
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(simple_test_nop2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(nop{});
    mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(nop{});
    run_pass(p);
    EXPECT(std::distance(mm->begin(), mm->end()) == 2);
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(duplicate_test1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(sum_op{}, one, two);
    auto count = std::distance(mm->begin(), mm->end());
    run_pass(p);
    EXPECT(std::distance(mm->begin(), mm->end()) == (count - 1));
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(duplicate_test2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(minus_op{}, one, two);
    mm->add_instruction(sum_op{}, one, two);
    auto count = std::distance(mm->begin(), mm->end());
    run_pass(p);
    EXPECT(std::distance(mm->begin(), mm->end()) == (count - 2));
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(depth_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto x1  = mm->add_instruction(sum_op{}, one, two);
    auto x2  = mm->add_instruction(sum_op{}, one, two);
    mm->add_instruction(minus_op{}, x1, x2);
    mm->add_instruction(minus_op{}, x1, x2);
    mm->add_instruction(sum_op{}, one, two);
    auto count = std::distance(mm->begin(), mm->end());
    run_pass(p);
    EXPECT(std::distance(mm->begin(), mm->end()) == (count - 4));
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(undefined_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(migraphx::make_op("undefined"));
    mm->add_instruction(sum_op{}, one, two);
    auto count = std::distance(mm->begin(), mm->end());
    run_pass(p);
    EXPECT(std::distance(mm->begin(), mm->end()) == count - 1);
    EXPECT(
        std::none_of(mm->begin(), mm->end(), [](auto&& ins) { return ins.name() == "undefined"; }));
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(duplicate_args1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_literal(0);
    auto l3  = mm->add_literal(3);
    mm->add_instruction(migraphx::make_op("add"), l3, l3);
    mm->add_instruction(migraphx::make_op("identity"), l0);
    auto count = std::distance(mm->begin(), mm->end());
    run_pass(p);
    EXPECT(std::distance(mm->begin(), mm->end()) != count);
    EXPECT(std::distance(mm->begin(), mm->end()) == 2);
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{0});
}

TEST_CASE(duplicate_args2)
{
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto l0   = mm->add_literal(0);
    auto l3   = mm->add_literal(3);
    auto sum1 = mm->add_instruction(migraphx::make_op("add"), l0, l3);
    mm->add_instruction(migraphx::make_op("add"), sum1, l3);
    mm->add_instruction(migraphx::make_op("identity"), l0);
    auto count = std::distance(mm->begin(), mm->end());
    run_pass(p);
    EXPECT(std::distance(mm->begin(), mm->end()) != count);
    EXPECT(std::distance(mm->begin(), mm->end()) == 2);
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{0});
}

TEST_CASE(duplicate_args3)
{
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto l0   = mm->add_literal(0);
    auto l3   = mm->add_literal(3);
    auto sum1 = mm->add_instruction(migraphx::make_op("add"), l0, l3);
    auto sum2 = mm->add_instruction(migraphx::make_op("add"), l0, sum1);
    mm->add_instruction(migraphx::make_op("add"), sum2, l3);
    mm->add_instruction(migraphx::make_op("identity"), l0);
    auto count = std::distance(mm->begin(), mm->end());
    run_pass(p);
    EXPECT(std::distance(mm->begin(), mm->end()) != count);
    EXPECT(std::distance(mm->begin(), mm->end()) == 2);
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{0});
}

TEST_CASE(reused_twice)
{
    migraphx::program p;
    auto* mm                 = p.get_main_module();
    std::vector<size_t> dims = {1, 2, 2};
    auto x        = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, dims});
    auto y        = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type, dims});
    auto z        = mm->add_parameter("z", migraphx::shape{migraphx::shape::float_type, dims});
    auto add1     = mm->add_instruction(migraphx::make_op("add"), x, y);
    auto add2     = mm->add_instruction(migraphx::make_op("add"), add1, z);
    auto epsilon  = mm->add_literal(1e-12f);
    auto exponent = mm->add_literal(2.0f);

    auto mean = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), add2);
    auto mean_mbcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", dims}}), mean);
    auto sub = mm->add_instruction(migraphx::make_op("sub"), add2, mean_mbcast);
    auto exponent_mbcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", dims}}), exponent);
    auto pow = mm->add_instruction(migraphx::make_op("pow"), sub, exponent_mbcast);
    auto var = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), pow);
    auto epsilon_mbcast = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {1, dims.at(1), 1}}}), epsilon);
    auto add_epsilon = mm->add_instruction(migraphx::make_op("add"), var, epsilon_mbcast);
    mm->add_instruction(migraphx::make_op("sqrt"), add_epsilon);
    mm->add_instruction(migraphx::make_op("add"), x, y);

    auto count = std::distance(mm->begin(), mm->end());
    run_pass(p);
    EXPECT(std::distance(mm->begin(), mm->end()) != count);
    EXPECT(std::distance(mm->begin(), mm->end()) == 4);
}

TEST_CASE(unused_module)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto* m1 = p.create_module("unused");
    auto* m2 = p.create_module("used");
    auto l0  = mm->add_literal(0);
    m1->add_literal(0);
    m2->add_literal(0);
    mm->add_instruction(mod_pass_op{}, {l0}, {m2});
    EXPECT(migraphx::contains(p.get_modules(), m1));
    EXPECT(migraphx::contains(p.get_modules(), m2));
    run_pass(p);
    EXPECT(migraphx::contains(p.get_modules(), m2));
    EXPECT(not migraphx::contains(p.get_modules(), m1));
}

TEST_CASE(param_not_eliminated)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::int32_type, {2, 2}};
        auto x = mm->add_parameter("x", s);
        auto y = mm->add_parameter("y", s);
        mm->add_parameter("z", s);
        auto sum = mm->add_instruction(migraphx::make_op("add"), x, y);
        mm->add_return({sum});

        return p;
    };

    auto p = create_program();
    run_pass(p);
    EXPECT(p == create_program());
}

TEST_CASE(eliminate_tuple_ins)
{
    auto create_program = [](bool add_dead_ins) {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sd{migraphx::shape::float_type, {1}};
        auto l1 = mm->add_literal(migraphx::literal(sd, {1}));
        auto l2 = mm->add_literal(migraphx::literal(sd, {2}));
        auto l3 = mm->add_literal(migraphx::literal(sd, {3}));
        migraphx::shape sx{migraphx::shape::float_type, {1, 4}};
        migraphx::shape sy{migraphx::shape::float_type, {3, 4}};
        migraphx::shape sc{migraphx::shape::bool_type};
        auto cond = mm->add_parameter("cond", sc);
        auto x    = mm->add_parameter("x", sx);
        auto y    = mm->add_parameter("y", sy);

        auto* then_mod = p.create_module("If_6_if");
        auto m1        = then_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 4}}}), l1);
        auto add0 = then_mod->add_instruction(migraphx::make_op("add"), x, m1);
        auto m2   = then_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {3, 4}}}), l2);
        auto mul0  = then_mod->add_instruction(migraphx::make_op("mul"), y, m2);
        auto mfp16 = then_mod->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), mul0);
        then_mod->add_return({add0, mul0, mfp16});

        auto* else_mod = p.create_module("If_6_else");
        auto me1       = else_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 4}}}), l3);
        auto mul1 = else_mod->add_instruction(migraphx::make_op("mul"), x, me1);
        auto me2  = else_mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {3, 4}}}), l3);
        auto add1  = else_mod->add_instruction(migraphx::make_op("add"), y, me2);
        auto afp16 = else_mod->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), add1);
        else_mod->add_return({mul1, add1, afp16});
        if(add_dead_ins)
        {
            auto neg_cond = mm->add_instruction(migraphx::make_op("neg"), {cond});
            mm->add_instruction(migraphx::make_op("if"), {neg_cond}, {then_mod, else_mod});
        }
        auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        auto r0  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
        auto r1  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), ret);
        auto r16 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), ret);
        mm->add_return({r0, r1, r16});

        return p;
    };
    auto p = create_program(true);
    run_pass(p);
    EXPECT(p == create_program(false));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
