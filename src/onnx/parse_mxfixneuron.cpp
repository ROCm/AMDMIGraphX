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
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/tune_axis.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_mxfixneuron : op_parser<parse_mxfixneuron>
{
    std::vector<op_desc> operators() const { return {{"MXFixNeuron"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        const instruction_ref input = args.front();
        instruction_ref tmp_in      = input;
        const auto input_lens = input->get_shape().lens();
        if(args.size() != 1)
        {
            MIGRAPHX_THROW("MXFixNeuron: must have only 1 input");
        }
        int block_axis = info.attributes.at("axis").i();
        block_axis     = tune_axis(input->get_shape().ndim(), block_axis, "MXFixNeuron");
        int block_size = info.attributes.at("block_size").i();
        if(block_size != 32)
        {
            MIGRAPHX_THROW("MXFixNeuron: only block_size of 32 is supported");
        }
        std::string element_dtype = info.attributes.at("element_dtype").s();
        if(element_dtype != "fp4_e2m1")
        {
            MIGRAPHX_THROW("MXFixNeuron: only support MXFP4 type");
        }
        int rounding_mode = info.attributes.at("rounding_mode").i();
        if(rounding_mode != 2)
        {
            MIGRAPHX_THROW("MXFixNeuron: only round ties to even is supported");
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

        // broadcast scales for use in quantizelinear
        block_scales_ins = info.add_instruction(
            make_op("multibroadcast", {{"out_lens", reduct_dims}}), block_scales_ins);
        block_scales_ins =
            info.add_instruction(make_op("reshape", {{"dims", tmp_lens}}), block_scales_ins);

        // if padded runt block do slicing
        if(tmp_lens != input_lens)
        {
            std::size_t slice_size = input_lens.at(block_axis);
            block_scales_ins       = info.add_instruction(
                make_op("slice", {{"axes", {block_axis}}, {"starts", {0}}, {"ends", {slice_size}}}),
                block_scales_ins);
        }

        auto q_ins = info.add_instruction(
            make_op("quantizelinear", {{"out_type", migraphx::shape::float_type}}),
            input,
            block_scales_ins); // output is float_type

        // ravel tensor to 1D for handling possible odd number of elements for packing
        auto quantized_shape     = q_ins->get_shape();
        std::size_t num_elements = quantized_shape.elements();
        auto ravel_ins =
            info.add_instruction(make_op("reshape", {{"dims", {num_elements}}}), q_ins);
        bool odd_num_elem = (num_elements % 2 == 1);
        if(odd_num_elem)
        {
            // pad one element at end if odd number of elements
            ravel_ins = info.add_instruction(make_op("pad", {{"pads", {0, 1}}}), ravel_ins);
        }
        auto pack_ins =
            info.add_instruction(make_op("pack_fp4"), ravel_ins); // output is packed_fp4_type
        auto unpack_ins =
            info.add_instruction(make_op("unpack_fp4"), pack_ins); // output is float_type
        if(odd_num_elem)
        {
            // slice off padded value
            unpack_ins = info.add_instruction(
                make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {num_elements}}}),
                unpack_ins);
        }
        auto reshape_unpack_ins = info.add_instruction(
            make_op("reshape", {{"dims", quantized_shape.lens()}}), unpack_ins);
        auto dq_ins =
            info.add_instruction(make_op("dequantizelinear"), reshape_unpack_ins, block_scales_ins);
        return dq_ins;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
