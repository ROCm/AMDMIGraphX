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
#include <migraphx/argument.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/pad.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>

#include <test.hpp>

TEST_CASE(pad_border_constant)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    migraphx::shape bs{migraphx::shape::float_type, {4, 4}};
    auto x      = mm->add_literal(migraphx::literal{s, {1, 2, 3, 4}});
    auto buffer = mm->add_parameter("buffer", bs);
    mm->add_instruction(
        migraphx::make_op("pad_border", {{"pads", {1, 1, 1, 1}}, {"value", 9}}), x, buffer);
    p.compile(migraphx::make_target("ref"));
    // The interior is left untouched
    std::vector<float> data(16, -1);
    migraphx::parameter_map params;
    params["buffer"] = migraphx::argument(bs, data.data());
    auto result      = p.eval(params).back();
    std::vector<float> gold{9, 9, 9, 9, 9, -1, -1, 9, 9, -1, -1, 9, 9, 9, 9, 9};
    EXPECT(result.to_vector<float>() == gold);
}

TEST_CASE(pad_border_constant_asym)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    migraphx::shape bs{migraphx::shape::float_type, {4, 3}};
    auto x      = mm->add_literal(migraphx::literal{s, {1, 2, 3, 4}});
    auto buffer = mm->add_parameter("buffer", bs);
    mm->add_instruction(migraphx::make_op("pad_border", {{"pads", {0, 1, 2, 0}}}), x, buffer);
    p.compile(migraphx::make_target("ref"));
    std::vector<float> data(12, -1);
    migraphx::parameter_map params;
    params["buffer"] = migraphx::argument(bs, data.data());
    auto result      = p.eval(params).back();
    std::vector<float> gold{0, -1, -1, 0, -1, -1, 0, 0, 0, 0, 0, 0};
    EXPECT(result.to_vector<float>() == gold);
}

TEST_CASE(pad_border_reflect)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    migraphx::shape bs{migraphx::shape::float_type, {4, 4}};
    auto x      = mm->add_literal(migraphx::literal{s, {1, 2, 3, 4}});
    auto buffer = mm->add_parameter("buffer", bs);
    mm->add_instruction(
        migraphx::make_op("pad_border",
                          {{"pads", {1, 1, 1, 1}}, {"mode", migraphx::op::pad::reflect_pad}}),
        x,
        buffer);
    p.compile(migraphx::make_target("ref"));
    // The border is read from the interior given as input, not from the buffer
    std::vector<float> data(16, -1);
    migraphx::parameter_map params;
    params["buffer"] = migraphx::argument(bs, data.data());
    auto result      = p.eval(params).back();
    std::vector<float> gold{4, 3, 4, 3, 2, -1, -1, 1, 4, -1, -1, 3, 2, 1, 2, 1};
    EXPECT(result.to_vector<float>() == gold);
}

TEST_CASE(pad_border_edge)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    migraphx::shape bs{migraphx::shape::float_type, {4, 4}};
    auto x      = mm->add_literal(migraphx::literal{s, {1, 2, 3, 4}});
    auto buffer = mm->add_parameter("buffer", bs);
    mm->add_instruction(
        migraphx::make_op("pad_border",
                          {{"pads", {1, 1, 1, 1}}, {"mode", migraphx::op::pad::edge_pad}}),
        x,
        buffer);
    p.compile(migraphx::make_target("ref"));
    std::vector<float> data(16, -1);
    migraphx::parameter_map params;
    params["buffer"] = migraphx::argument(bs, data.data());
    auto result      = p.eval(params).back();
    std::vector<float> gold{1, 1, 2, 2, 1, -1, -1, 2, 3, -1, -1, 4, 3, 3, 4, 4};
    EXPECT(result.to_vector<float>() == gold);
}
