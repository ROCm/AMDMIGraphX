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

#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/tune_axis.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

/**
 * Operator from Brevitas to calculate dynamic quantization scales.
 */
struct parse_dynamicscale : op_parser<parse_dynamicscale>
{

    std::vector<op_desc> operators() const { return {{"DynamicScale"}}; };

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          onnx_parser::node_info info,
                          const std::vector<instruction_ref>& args) const
    {
        const instruction_ref input = args.front();
        instruction_ref tmp_in      = input;
        const auto input_lens       = input->get_shape().lens();
        if(args.size() != 1)
        {
            MIGRAPHX_THROW("DynamicScale: must have only 1 input");
        }
        int block_axis = info.attributes.at("group_dim").i();
        block_axis     = tune_axis(input->get_shape().ndim(), block_axis, "DynamicScale");
        int block_size = info.attributes.at("group_size").i();
        if(block_size != 32)
        {
            MIGRAPHX_THROW("DynamicScale: only group_size of 32 is supported");
        }
        migraphx::shape::type_t output_type = get_type(info.attributes.at("output_dtype").i());

        // TODO expand this to handle other MX types
        if(output_type != migraphx::shape::fp4x2_type)
        {
            MIGRAPHX_THROW("DynamicScale: only support MXFP4 type");
        }

        std::string scale_selection_method = info.attributes.at("scale_selection_method").s();
        if(scale_selection_method != "floor")
        {
            MIGRAPHX_THROW("DynamicScale: only support floor scale selection");
        }

        std::string zero_point_selection_method = "None";
        if(contains(info.attributes, "zero_point_selection_method"))
            zero_point_selection_method = info.attributes.at("zero_point_selection_method").s();

        if(zero_point_selection_method != "None")
        {
            MIGRAPHX_THROW("DynamicScale: zero_point not supported");
        }

        // make reduction axes for calculating block scales
        // tmp_lens != input_lens if runt block is padded
        auto tmp_lens  = input_lens;
        auto block_dim = tmp_lens.at(block_axis);
        std::size_t block_padding =
            std::ceil(double(block_dim) / double(block_size)) * block_size - block_dim;
        // handle runt block by padding
        if(block_padding != 0)
        {
            std::vector<std::size_t> pads_vec(2 * tmp_lens.size(), 0);
            pads_vec.at(block_axis + tmp_lens.size()) = block_padding;
            tmp_in   = info.add_instruction(make_op("pad", {{"pads", pads_vec}}), tmp_in);
            tmp_lens = tmp_in->get_shape().lens();
        }
        // reshape block dimension to {num_blocks, block_size}
        std::size_t num_blocks               = tmp_lens.at(block_axis) / std::size_t(block_size);
        std::vector<std::size_t> reduct_dims = tmp_lens;
        reduct_dims.at(block_axis)           = block_size;
        reduct_dims.insert(reduct_dims.begin() + block_axis, num_blocks);
        instruction_ref reshape_ins =
            info.add_instruction(make_op("reshape", {{"dims", reduct_dims}}), tmp_in);

        // dynamic quantization for MX types:
        // V_k = fp32 vector input of block size k
        // B_k = pow(2, floor(log2(reduce_max(abs(V_k))))) # largest power of 2 less than V
        // X_k = block scale k = B_k / (largest power of 2 in fp4e2m1) = B_k / 4
        auto abs_ins = info.add_instruction(make_op("abs"), reshape_ins);
        auto reduce_max_ins =
            info.add_instruction(make_op("reduce_max", {{"axes", {block_axis + 1}}}), abs_ins);
        auto log2_ins  = info.add_instruction(make_op("log2"), reduce_max_ins);
        auto floor_ins = info.add_instruction(make_op("floor"), log2_ins);
        auto lit_2_ins = info.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {2.f}});
        auto broadcast_lit_2_ins = info.add_instruction(
            make_op("multibroadcast", {{"out_lens", reduce_max_ins->get_shape().lens()}}),
            lit_2_ins);
        auto pow_ins   = info.add_instruction(make_op("pow"), broadcast_lit_2_ins, floor_ins);
        auto lit_4_ins = info.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {4.f}});
        auto broadcast_lit_4_ins = info.add_instruction(
            make_op("multibroadcast", {{"out_lens", reduce_max_ins->get_shape().lens()}}),
            lit_4_ins);
        auto block_scales_ins = info.add_instruction(make_op("div"), pow_ins, broadcast_lit_4_ins);

        // squeeze reduction axis for use in block quantized quantizelinear
        block_scales_ins = info.add_instruction(make_op("squeeze", {{"axes", {block_axis + 1}}}),
                                                block_scales_ins);

        return block_scales_ins;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
