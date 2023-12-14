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

#include <onnx_test.hpp>

TEST_CASE(resize_upsample_linear_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape ss{migraphx::shape::float_type, {4}};
    std::vector<float> ds = {1, 1, 2, 2};
    mm->add_literal(migraphx::literal(ss, ds));

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    auto x = mm->add_parameter("X", sx);
    migraphx::shape s_ind{migraphx::shape::int32_type, {16, 1, 4, 4}};
    std::vector<int> d_ind = {
        0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 2,
        2, 2, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
        0, 1, 2, 2, 2, 3, 0, 0, 0, 1, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 0, 0, 0, 1, 2, 2, 2,
        3, 2, 2, 2, 3, 2, 2, 2, 3, 0, 0, 0, 1, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 0, 0, 0, 1,
        2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 2, 3, 3, 3, 0,
        1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 2, 3, 3, 3, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 2, 3,
        3, 3, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 2, 3, 3, 3, 0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3,
        3, 2, 3, 3, 3, 0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 0, 1, 1, 1, 2, 3, 3, 3,
        2, 3, 3, 3, 2, 3, 3, 3, 0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3};
    auto l_ind = mm->add_literal(migraphx::literal(s_ind, d_ind));

    migraphx::shape s8{migraphx::shape::float_type, {8, 1, 4, 4}};
    std::vector<float> d8 = {
        0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0,
        0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0,
        0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0,
        0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0,
        0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0,
        0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0,
        0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0,
        0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0,
        0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0,
        0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0,
        0, 1.0f / 3, 2.0f / 3, 0, 0, 1.0f / 3, 2.0f / 3, 0};
    auto l8 = mm->add_literal(migraphx::literal(s8, d8));

    migraphx::shape s4{migraphx::shape::float_type, {4, 1, 4, 4}};
    std::vector<float> d4 = {
        0,        0,        0,        0,        1.0f / 3, 1.0f / 3, 1.0f / 3, 1.0f / 3,
        2.0f / 3, 2.0f / 3, 2.0f / 3, 2.0f / 3, 0,        0,        0,        0,
        0,        0,        0,        0,        1.0f / 3, 1.0f / 3, 1.0f / 3, 1.0f / 3,
        2.0f / 3, 2.0f / 3, 2.0f / 3, 2.0f / 3, 0,        0,        0,        0,
        0,        0,        0,        0,        1.0f / 3, 1.0f / 3, 1.0f / 3, 1.0f / 3,
        2.0f / 3, 2.0f / 3, 2.0f / 3, 2.0f / 3, 0,        0,        0,        0,
        0,        0,        0,        0,        1.0f / 3, 1.0f / 3, 1.0f / 3, 1.0f / 3,
        2.0f / 3, 2.0f / 3, 2.0f / 3, 2.0f / 3, 0,        0,        0,        0};
    auto l4 = mm->add_literal(migraphx::literal(s4, d4));

    migraphx::shape s2{migraphx::shape::float_type, {2, 1, 4, 4}};
    std::vector<float> d2(32, 0);
    auto l2 = mm->add_literal(migraphx::literal(s2, d2));

    migraphx::shape s1{migraphx::shape::float_type, {1, 1, 4, 4}};
    std::vector<float> d1(16, 0.0f);
    auto l1 = mm->add_literal(migraphx::literal(s1, d1));

    mm->add_instruction(migraphx::make_op("undefined"));
    auto rsp   = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), x);
    auto data  = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), rsp, l_ind);
    auto slc80 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {8}}}), data);
    auto slc81 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {8}}, {"ends", {16}}}), data);
    auto diff8 = mm->add_instruction(migraphx::make_op("sub"), slc81, slc80);
    auto mul8  = mm->add_instruction(migraphx::make_op("mul"), diff8, l8);
    auto add8  = mm->add_instruction(migraphx::make_op("add"), mul8, slc80);
    auto slc40 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {4}}}), add8);
    auto slc41 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {4}}, {"ends", {8}}}), add8);
    auto diff4 = mm->add_instruction(migraphx::make_op("sub"), slc41, slc40);
    auto mul4  = mm->add_instruction(migraphx::make_op("mul"), diff4, l4);
    auto add4  = mm->add_instruction(migraphx::make_op("add"), mul4, slc40);
    auto slc20 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {2}}}), add4);
    auto slc21 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {4}}}), add4);
    auto diff2 = mm->add_instruction(migraphx::make_op("sub"), slc21, slc20);
    auto mul2  = mm->add_instruction(migraphx::make_op("mul"), diff2, l2);
    auto add2  = mm->add_instruction(migraphx::make_op("add"), mul2, slc20);
    auto slc10 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), add2);
    auto slc11 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), add2);
    auto diff1 = mm->add_instruction(migraphx::make_op("sub"), slc11, slc10);
    auto mul1  = mm->add_instruction(migraphx::make_op("mul"), diff1, l1);
    auto add1  = mm->add_instruction(migraphx::make_op("add"), mul1, slc10);
    mm->add_return({add1});

    auto prog = migraphx::parse_onnx("resize_upsample_linear_test.onnx");
    EXPECT(p == prog);
}
