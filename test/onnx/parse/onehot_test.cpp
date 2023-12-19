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

#include <onnx_test.hpp>

TEST_CASE(onehot_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s_ind{migraphx::shape::int32_type, {5, 2}};
    migraphx::shape s_val{migraphx::shape::half_type, {2}};
    mm->add_literal(3);
    auto l_ind = mm->add_parameter("indices", s_ind);
    auto l_val = mm->add_parameter("values", s_val);
    migraphx::shape s_dep{migraphx::shape::half_type, {3, 3}};
    std::vector<float> data_dep{1, 0, 0, 0, 1, 0, 0, 0, 1};
    auto l_dep      = mm->add_literal(migraphx::literal(s_dep, data_dep));
    auto gather_out = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), l_dep, l_ind);
    auto tr_out  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 0, 1}}}),
                                      gather_out);
    auto off_val = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), l_val);
    auto on_val = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), l_val);
    auto diff       = mm->add_instruction(migraphx::make_op("sub"), on_val, off_val);
    auto mb_off_val = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {3, 5, 2}}}), off_val);
    auto mb_diff =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 5, 2}}}), diff);
    auto mul = mm->add_instruction(migraphx::make_op("mul"), tr_out, mb_diff);
    auto r   = mm->add_instruction(migraphx::make_op("add"), mul, mb_off_val);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("onehot_test.onnx");

    EXPECT(p == prog);
}
