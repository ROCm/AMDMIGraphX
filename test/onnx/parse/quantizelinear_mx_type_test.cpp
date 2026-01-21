/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <onnx_test_utils.hpp>

// even fastest dimension
TEST_CASE(quantizelinear_mxfp4_even_test)
{
    migraphx::program p;
    auto* mm        = p.get_main_module();
    auto l0         = mm->add_parameter("0", {migraphx::shape::float_type, {3, 64, 4, 4}});
    auto l1         = mm->add_parameter("1", {migraphx::shape::float_type, {3, 2, 4, 4}});
    auto l1_reshape = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), l1);
    l1_reshape      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {3, 2, 32, 4, 4}}}), l1_reshape);
    l1_reshape =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {3, 64, 4, 4}}}), l1_reshape);
    auto q_ins = mm->add_instruction(
        migraphx::make_op("quantizelinear", {{"out_type", migraphx::shape::float_type}}),
        l0,
        l1_reshape);
    auto pack_ins   = mm->add_instruction(migraphx::make_op("pack_fp4", {{"axis", 3}}), q_ins);
    auto unpack_ins = mm->add_instruction(migraphx::make_op("unpack_fp4", {{"axis", 3}}), pack_ins);
    mm->add_return({unpack_ins});

    auto prog = read_onnx("quantizelinear_mxfp4_even_test.onnx");
    EXPECT(p.sort() == prog.sort());
}

// odd fastest dimension
TEST_CASE(quantizelinear_mxfp4_odd_test)
{
    migraphx::program p;
    auto* mm        = p.get_main_module();
    auto l0         = mm->add_parameter("0", {migraphx::shape::float_type, {3, 64, 4, 7}});
    auto l1         = mm->add_parameter("1", {migraphx::shape::float_type, {3, 2, 4, 7}});
    auto l1_reshape = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), l1);
    l1_reshape      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {3, 2, 32, 4, 7}}}), l1_reshape);
    l1_reshape =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {3, 64, 4, 7}}}), l1_reshape);
    auto q_ins = mm->add_instruction(
        migraphx::make_op("quantizelinear", {{"out_type", migraphx::shape::float_type}}),
        l0,
        l1_reshape);
    auto pad_ins =
        mm->add_instruction(migraphx::make_op("pad", {{"pads", {0, 0, 0, 0, 0, 0, 0, 1}}}), q_ins);
    auto pack_ins   = mm->add_instruction(migraphx::make_op("pack_fp4", {{"axis", 3}}), pad_ins);
    auto unpack_ins = mm->add_instruction(migraphx::make_op("unpack_fp4", {{"axis", 3}}), pack_ins);
    auto slice_ins  = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {7}}}), unpack_ins);
    mm->add_return({slice_ins});

    auto prog = read_onnx("quantizelinear_mxfp4_odd_test.onnx");
    EXPECT(p.sort() == prog.sort());
}
