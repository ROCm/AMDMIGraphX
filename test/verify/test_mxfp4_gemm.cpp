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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>

#include <cmath>

namespace {
migraphx::instruction_ref add_dyn_scale_calc(migraphx::module_ref m,
                                             migraphx::instruction_ref input,
                                             int block_axis,
                                             int block_size)
{
    using migraphx::instruction_ref;
    using migraphx::make_op;
    using migraphx::module_ref;

    // Code similar to that in parse_mxqdq
    // make reduction axes for calculating block scales
    // tmp_lens != input_lens if runt block is padded
    instruction_ref tmp_in = input;
    const auto input_lens  = input->get_shape().lens();
    auto tmp_lens          = input_lens;
    auto block_dim         = tmp_lens.at(block_axis);
    std::size_t block_padding =
        std::ceil(double(block_dim) / double(block_size)) * block_size - block_dim;
    // handle runt block by padding
    if(block_padding != 0)
    {
        std::vector<std::size_t> pads_vec(2 * tmp_lens.size(), 0);
        pads_vec.at(block_axis + tmp_lens.size()) = block_padding;
        tmp_in   = m->add_instruction(make_op("pad", {{"pads", pads_vec}}), tmp_in);
        tmp_lens = tmp_in->get_shape().lens();
    }
    // reshape block dimension to {num_blocks, block_size}
    std::size_t num_blocks               = tmp_lens.at(block_axis) / std::size_t(block_size);
    std::vector<std::size_t> reduct_dims = tmp_lens;
    reduct_dims.at(block_axis)           = block_size;
    reduct_dims.insert(reduct_dims.begin() + block_axis, num_blocks);
    instruction_ref reshape_ins =
        m->add_instruction(make_op("reshape", {{"dims", reduct_dims}}), tmp_in);

    // dynamic quantization for MX types:
    // V_k = fp32 vector input of block size k
    // B_k = pow(2, floor(log2(reduce_max(abs(V_k))))) # largest power of 2 less than V
    // X_k = block scale k = B_k / (largest power of 2 in fp4e2m1) = B_k / 4
    auto abs_ins = m->add_instruction(make_op("abs"), reshape_ins);
    auto reduce_max_ins =
        m->add_instruction(make_op("reduce_max", {{"axes", {block_axis + 1}}}), abs_ins);
    auto log2_ins  = m->add_instruction(make_op("log2"), reduce_max_ins);
    auto floor_ins = m->add_instruction(make_op("floor"), log2_ins);
    auto lit_2_ins =
        m->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {2.f}});
    auto broadcast_lit_2_ins = m->add_instruction(
        make_op("multibroadcast", {{"out_lens", reduce_max_ins->get_shape().lens()}}), lit_2_ins);
    auto pow_ins = m->add_instruction(make_op("pow"), broadcast_lit_2_ins, floor_ins);
    auto lit_4_ins =
        m->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {4.f}});
    auto broadcast_lit_4_ins = m->add_instruction(
        make_op("multibroadcast", {{"out_lens", reduce_max_ins->get_shape().lens()}}), lit_4_ins);
    auto block_scales_ins = m->add_instruction(make_op("div"), pow_ins, broadcast_lit_4_ins);

    // broadcast scales for use in quantizelinear
    block_scales_ins = m->add_instruction(make_op("multibroadcast", {{"out_lens", reduct_dims}}),
                                          block_scales_ins);
    block_scales_ins =
        m->add_instruction(make_op("reshape", {{"dims", tmp_lens}}), block_scales_ins);

    // if padded runt block do slicing
    if(tmp_lens != input_lens)
    {
        std::size_t slice_size = input_lens.at(block_axis);
        block_scales_ins       = m->add_instruction(
            make_op("slice", {{"axes", {block_axis}}, {"starts", {0}}, {"ends", {slice_size}}}),
            block_scales_ins);
    }
    return block_scales_ins;
}
} // namespace

/**
 * Designed to be like the final GEMM of resnet50.
 */
struct test_mxfp4_gemm : verify_program<test_mxfp4_gemm>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::module_ref mmain = p.get_main_module();
        auto input =
            mmain->add_parameter("x1", migraphx::shape{migraphx::shape::float_type, {1, 2048}});
        auto input_scales = add_dyn_scale_calc(mmain, input, 1, 32);
        input             = mmain->add_instruction(
            migraphx::make_op("quantizelinear", {{"out_type", migraphx::shape::float_type}}),
            input,
            input_scales);
        input = mmain->add_instruction(migraphx::make_op("pack_fp4"), input);
        input = mmain->add_instruction(migraphx::make_op("unpack_fp4"), input);
        input = mmain->add_instruction(migraphx::make_op("dequantizelinear"), input, input_scales);
        auto weights       = mmain->add_literal(migraphx::generate_literal(
            migraphx::shape{migraphx::shape::float_type, {1000, 2048}}, 2));
        auto weight_scales = add_dyn_scale_calc(mmain, weights, 1, 32);
        weights            = mmain->add_instruction(
            migraphx::make_op("quantizelinear", {{"out_type", migraphx::shape::float_type}}),
            weights,
            weight_scales);
        weights = mmain->add_instruction(migraphx::make_op("pack_fp4"), weights);
        weights = mmain->add_instruction(migraphx::make_op("unpack_fp4"), weights);
        weights =
            mmain->add_instruction(migraphx::make_op("dequantizelinear"), weights, weight_scales);
        weights = mmain->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}),
                                         weights);
        auto bias = mmain->add_literal(
            migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1, 1000}}, 0));
        auto dot      = mmain->add_instruction(migraphx::make_op("dot"), input, weights);
        auto bias_add = mmain->add_instruction(migraphx::make_op("add"), dot, bias);
        mmain->add_return({bias_add});
        return p;
    }
    std::string section() const { return "gemm"; }

    std::size_t get_tolerance() const { return 4e5; };
};
