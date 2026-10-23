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
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

// make_tuple packs its inputs without copying: each element is the input's
// own buffer
TEST_CASE(make_tuple_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::float_type, {2, 2}};
    migraphx::shape s1{migraphx::shape::int32_type, {3}};
    std::vector<float> d0{1, 2, 3, 4};
    std::vector<int> d1{5, 6, 7};
    auto l0    = mm->add_literal(migraphx::literal{s0, d0});
    auto l1    = mm->add_literal(migraphx::literal{s1, d1});
    auto tuple = mm->add_instruction(migraphx::make_op("make_tuple"), l0, l1);
    EXPECT(tuple->get_shape() == migraphx::shape({s0, s1}));
    EXPECT(migraphx::instruction::get_output_alias(tuple) ==
           std::vector<migraphx::instruction_ref>{l0, l1});
    auto e0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), tuple);
    auto e1 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), tuple);
    mm->add_return({e0, e1});
    p.compile(migraphx::make_target("ref"));
    auto results = p.eval({});
    EXPECT(results.size() == 2);
    std::vector<float> r0;
    results[0].visit([&](auto output) { r0.assign(output.begin(), output.end()); });
    std::vector<int> r1;
    results[1].visit([&](auto output) { r1.assign(output.begin(), output.end()); });
    EXPECT(r0 == d0);
    EXPECT(r1 == d1);
}
