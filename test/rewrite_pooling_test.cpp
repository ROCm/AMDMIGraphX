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
#include <migraphx/rewrite_pooling.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/ranges.hpp>
#include <test.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/verify.hpp>

bool is_pooling(migraphx::instruction& ins) { return ins.name() == "pooling"; }
static void opt_pooling(migraphx::module& m)
{
    migraphx::rewrite_pooling rp;
    migraphx::dead_code_elimination dce;
    rp.apply(m);
    dce.apply(m);
}

TEST_CASE(rewrite_pooling_test)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3, 4, 5}};
    auto pooling_program = [&](const migraphx::op::pooling_mode mode) {
        migraphx::module m;
        auto input = m.add_parameter("x", s);
        auto ret   = m.add_instruction(migraphx::make_op("pooling",
                                                         {{"mode", mode},
                                                          {"padding", {0, 0, 0}},
                                                          {"stride", {1, 1, 1}},
                                                          {"lengths", {3, 4, 5}}}),
                                     input);
        m.add_return({ret});
        return m;
    };

    auto opt_program = [&](const migraphx::operation& reduce_op) {
        migraphx::module m;
        auto input = m.add_parameter("x", s);
        auto rdm   = m.add_instruction(reduce_op, input);
        m.add_return({rdm});
        return m;
    };

    auto test_rewrite = [&](const migraphx::op::pooling_mode mode, const migraphx::operation& op) {
        migraphx::module m1 = pooling_program(mode);
        migraphx::module m2 = opt_program(op);
        opt_pooling(m1);
        EXPECT(m1 == m2);
    };

    test_rewrite(migraphx::op::pooling_mode::average,
                 migraphx::make_op("reduce_mean", {{"axes", {2, 3, 4}}}));
    test_rewrite(migraphx::op::pooling_mode::max,
                 migraphx::make_op("reduce_max", {{"axes", {2, 3, 4}}}));
}

TEST_CASE(rewrite_avepooling_na1_test)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3, 4, 5}};
    auto pooling_program = [&]() {
        migraphx::module m;

        auto input = m.add_parameter("x", s);
        auto ret =
            m.add_instruction(migraphx::make_op("pooling",
                                                {{"mode", migraphx::op::pooling_mode::average},
                                                 {"padding", {0, 1, 0}},
                                                 {"stride", {1, 1, 1}},
                                                 {"lengths", {3, 4, 5}}}),
                              input);
        m.add_return({ret});
        return m;
    };

    migraphx::module m1 = pooling_program();
    migraphx::module m2 = m1;

    opt_pooling(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(rewrite_avepooling_na2_test)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3, 4, 5}};
    auto pooling_program = [&]() {
        migraphx::module m;

        auto input = m.add_parameter("x", s);
        auto ret =
            m.add_instruction(migraphx::make_op("pooling",
                                                {{"mode", migraphx::op::pooling_mode::average},
                                                 {"padding", {0, 0, 0}},
                                                 {"stride", {1, 2, 1}},
                                                 {"lengths", {3, 4, 5}}}),
                              input);
        m.add_return({ret});
        return m;
    };

    migraphx::module m1 = pooling_program();
    migraphx::module m2 = m1;

    opt_pooling(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(rewrite_avepooling_na3_test)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3, 4, 5}};
    auto pooling_program = [&]() {
        migraphx::module m;

        auto input = m.add_parameter("x", s);
        auto ret   = m.add_instruction(migraphx::make_op("pooling",
                                                         {{"mode", migraphx::op::pooling_mode::max},
                                                          {"padding", {0, 0, 0}},
                                                          {"stride", {1, 1, 1}},
                                                          {"lengths", {3, 3, 5}}}),
                                     input);
        m.add_return({ret});
        return m;
    };

    migraphx::module m1 = pooling_program();
    migraphx::module m2 = m1;

    opt_pooling(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(literal_rewrite_pooling_test)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3, 4, 5}};
    std::vector<float> data(s.elements());
    std::iota(data.begin(), data.end(), 1.0f);

    auto pooling_program = [&](const migraphx::op::pooling_mode mode) {
        migraphx::program p;

        auto* mm   = p.get_main_module();
        auto input = mm->add_literal(migraphx::literal(s, data));
        auto ret   = mm->add_instruction(migraphx::make_op("pooling",
                                                           {{"mode", mode},
                                                            {"padding", {0, 0, 0}},
                                                            {"stride", {1, 1, 1}},
                                                            {"lengths", {3, 4, 5}}}),
                                       input);
        mm->add_return({ret});
        return p;
    };

    auto opt_program = [&](const migraphx::operation& op) {
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto input = mm->add_literal(migraphx::literal(s, data));
        auto rsp   = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, -1}}}), input);
        auto rdm   = mm->add_instruction(op, rsp);
        auto ret =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 1, 1, 1}}}), rdm);
        mm->add_return({ret});

        return p;
    };

    auto test_rewrite_pooling = [&](const migraphx::op::pooling_mode mode,
                                    const migraphx::operation& op) {
        migraphx::program p1 = pooling_program(mode);
        migraphx::program p2 = opt_program(op);
        p1.compile(migraphx::make_target("ref"));
        p2.compile(migraphx::make_target("ref"));
        auto result1 = p1.eval({}).back();
        auto result2 = p2.eval({}).back();
        visit_all(result1, result2)(
            [&](auto r1, auto r2) { EXPECT(migraphx::verify::verify_rms_range(r1, r2)); });
    };

    test_rewrite_pooling(migraphx::op::pooling_mode::max,
                         migraphx::make_op("reduce_max", {{"axes", {1}}}));
    test_rewrite_pooling(migraphx::op::pooling_mode::average,
                         migraphx::make_op("reduce_mean", {{"axes", {1}}}));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
