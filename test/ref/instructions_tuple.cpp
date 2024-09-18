/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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

TEST_CASE(instructions_tuple_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s1{migraphx::shape::float_type, {1, 1}};
    migraphx::shape s2{migraphx::shape::float_type, {1, 2}};
    migraphx::shape s3{migraphx::shape::float_type, {1, 4}};
    auto x  = mm->add_parameter("x", s1);
    auto y  = mm->add_parameter("y", s2);
    auto z  = mm->add_parameter("z", s3);
    auto t  = mm->add_instruction(migraphx::make_op("instructions_tuple"), x, y, z);
    auto r0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), t);
    auto r1 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), t);
    auto r2 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), t);
    mm->add_return({r0, r1, r2});
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;
    std::vector<float> data{1.0, 2.0, 3.0, 4.0};
    pp["x"]      = migraphx::argument(s1, data.data());
    pp["y"]      = migraphx::argument(s2, data.data());
    pp["z"]      = migraphx::argument(s3, data.data());
    auto results = p.eval(pp);

    std::vector<float> res0;
    std::vector<float> res1;
    std::vector<float> res2;
    results.front().visit([&](auto v) { res0.assign(v.begin(), v.end()); });
    results.at(1).visit([&](auto v) { res1.assign(v.begin(), v.end()); });
    results.back().visit([&](auto v) { res2.assign(v.begin(), v.end()); });

    EXPECT(res0 == std::vector<float>{data.begin(), data.begin() + 1});
    EXPECT(res1 == std::vector<float>{data.begin(), data.begin() + 2});
    EXPECT(res2 == std::vector<float>{data.begin(), data.begin() + 4});

    EXPECT(results.front().get_shape() == s1);
    EXPECT(results.at(1).get_shape() == s2);
    EXPECT(results.back().get_shape() == s3);
}
