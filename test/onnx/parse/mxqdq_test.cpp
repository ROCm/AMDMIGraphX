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

TEST_CASE(mxqdq_even_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto input =
        mm->add_parameter("input", migraphx::shape{migraphx::shape::float_type, {3, 64, 4, 4}});
    auto reduce_reshape =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {3, 2, 32, 4, 4}}}), input);
    auto abs_ins = mm->add_instruction(migraphx::make_op("abs"), reduce_reshape);
    auto reduce_max_ins =
        mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {2}}}), abs_ins);
    auto log2_ins        = mm->add_instruction(migraphx::make_op("log2"), reduce_max_ins);
    auto floor_ins       = mm->add_instruction(migraphx::make_op("floor"), log2_ins);
    auto lit_2_ins       = mm->add_literal({migraphx::shape{migraphx::shape::float_type}, {2.f}});
    auto broadcast_lit_2 = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", reduce_max_ins->get_shape().lens()}}),
        lit_2_ins);
    auto pow_ins   = mm->add_instruction(migraphx::make_op("pow"), broadcast_lit_2, floor_ins);
    auto lit_4_ins = mm->add_literal({migraphx::shape{migraphx::shape::float_type}, {4.f}});
    auto broadcast_lit_4 = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", reduce_max_ins->get_shape().lens()}}),
        lit_4_ins);
    auto block_scales_ins = mm->add_instruction(migraphx::make_op("div"), pow_ins, broadcast_lit_4);
    block_scales_ins      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {3, 2, 32, 4, 4}}}), block_scales_ins);
    block_scales_ins = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {3, 64, 4, 4}}}),
                                           block_scales_ins);
    auto q_ins       = mm->add_instruction(
        migraphx::make_op("quantizelinear", {{"out_type", migraphx::shape::float_type}}),
        input,
        block_scales_ins);
    auto quantized_shape     = q_ins->get_shape();
    auto pack_ins            = mm->add_instruction(migraphx::make_op("pack_fp4"), q_ins);
    auto unpack_ins          = mm->add_instruction(migraphx::make_op("unpack_fp4"), pack_ins);
    mm->add_instruction(migraphx::make_op("dequantizelinear"), unpack_ins, block_scales_ins);

    auto prog = optimize_onnx("mxqdq_even_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(mxqdq_odd_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto input =
        mm->add_parameter("input", migraphx::shape{migraphx::shape::float_type, {71, 5, 5}});
    auto padded_input =
        mm->add_instruction(migraphx::make_op("pad", {{"pads", {0, 0, 0, 25, 0, 0}}}), input);
    auto reduce_reshape =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {3, 32, 5, 5}}}), padded_input);
    auto abs_ins = mm->add_instruction(migraphx::make_op("abs"), reduce_reshape);
    auto reduce_max_ins =
        mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {1}}}), abs_ins);
    auto log2_ins        = mm->add_instruction(migraphx::make_op("log2"), reduce_max_ins);
    auto floor_ins       = mm->add_instruction(migraphx::make_op("floor"), log2_ins);
    auto lit_2_ins       = mm->add_literal({migraphx::shape{migraphx::shape::float_type}, {2.f}});
    auto broadcast_lit_2 = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", reduce_max_ins->get_shape().lens()}}),
        lit_2_ins);
    auto pow_ins   = mm->add_instruction(migraphx::make_op("pow"), broadcast_lit_2, floor_ins);
    auto lit_4_ins = mm->add_literal({migraphx::shape{migraphx::shape::float_type}, {4.f}});
    auto broadcast_lit_4 = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", reduce_max_ins->get_shape().lens()}}),
        lit_4_ins);
    auto block_scales_ins = mm->add_instruction(migraphx::make_op("div"), pow_ins, broadcast_lit_4);
    block_scales_ins      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {3, 32, 5, 5}}}), block_scales_ins);
    block_scales_ins =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {96, 5, 5}}}), block_scales_ins);
    block_scales_ins = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {71}}}),
        block_scales_ins);
    auto q_ins = mm->add_instruction(
        migraphx::make_op("quantizelinear", {{"out_type", migraphx::shape::float_type}}),
        input,
        block_scales_ins);
    auto quantized_shape     = q_ins->get_shape();
    auto pad_ins =
        mm->add_instruction(migraphx::make_op("pad", {{"pads", {0, 0, 0, 0, 0, 1}}}), q_ins);
    auto pack_ins   = mm->add_instruction(migraphx::make_op("pack_fp4"), pad_ins);
    auto unpack_ins = mm->add_instruction(migraphx::make_op("unpack_fp4"), pack_ins);
    auto slice_ins  = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {5}}}), unpack_ins);
    mm->add_instruction(migraphx::make_op("dequantizelinear"), slice_ins, block_scales_ins);

    auto prog = optimize_onnx("mxqdq_odd_test.onnx");
    EXPECT(p == prog);
}
