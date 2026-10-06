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

TEST_CASE(make_tuple_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::float_type, {2, 2}};
    migraphx::shape s1{migraphx::shape::int32_type, {3}};
    std::vector<float> data0{1, 2, 3, 4};
    std::vector<int> data1{5, 6, 7};
    auto l0 = mm->add_literal(migraphx::literal{s0, data0});
    auto l1 = mm->add_literal(migraphx::literal{s1, data1});
    auto t  = mm->add_instruction(migraphx::make_op("make_tuple"), l0, l1);
    auto e0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), t);
    auto e1 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), t);
    mm->add_return({e0, e1});
    p.compile(migraphx::make_target("ref"));
    auto results = p.eval({});

    std::vector<float> results0;
    results.at(0).visit([&](auto output) { results0.assign(output.begin(), output.end()); });
    EXPECT(results0 == data0);
    std::vector<int> results1;
    results.at(1).visit([&](auto output) { results1.assign(output.begin(), output.end()); });
    EXPECT(results1 == data1);
}
