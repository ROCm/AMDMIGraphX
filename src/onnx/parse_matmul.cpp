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
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/common.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_matmul : op_parser<parse_matmul>
{
    std::vector<op_desc> operators() const
    {
        return {{"MatMul", "dot"}, {"MatMulInteger", "quant_dot"}};
    }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto a0 = args[0];
        auto a1 = args[1];
        auto s0 = a0->get_shape();
        auto s1 = a1->get_shape();

        instruction_ref dot_res;
        bool is_a_prepended = false;
        bool is_b_appended  = false;
        if(s0.ndim() == 1)
        {
            is_a_prepended = true;
            a0             = info.add_instruction(make_op("unsqueeze", {{"axes", {0}}}), args[0]);
        }
        if(s1.ndim() == 1)
        {
            is_b_appended = true;
            a1            = info.add_instruction(make_op("unsqueeze", {{"axes", {1}}}), args[1]);
        }

        auto is_quant_dot = opd.op_name == "quant_dot";
        if(s0.dynamic() or s1.dynamic())
        {
            if(is_quant_dot)
            {
                MIGRAPHX_THROW("PARSE_MATMUL: dynamic MatMulInteger not supported");
            }
            auto s0_dds = a0->get_shape().to_dynamic().dyn_dims();
            auto s1_dds = a1->get_shape().to_dynamic().dyn_dims();

            if(not std::equal(
                   s0_dds.rbegin() + 2, s0_dds.rend(), s1_dds.rbegin() + 2, s1_dds.rend()))
            {
                auto broadcasted_a0 = info.add_instruction(make_op("dot_broadcast"), a0, a1);
                auto broadcasted_a1 = info.add_instruction(make_op("dot_broadcast"), a1, a0);
                dot_res =
                    info.add_instruction(make_op(opd.op_name), broadcasted_a0, broadcasted_a1);
            }
            else
            {
                dot_res = info.add_instruction(make_op(opd.op_name), a0, a1);
            }
        }
        else
        {
            auto s0_lens        = a0->get_shape().lens();
            auto s1_lens        = a1->get_shape().lens();
            instruction_ref ba0 = a0;
            instruction_ref ba1 = a1;
            // try broadcasting if dimensions other than last two do not match
            if(not std::equal(
                   s0_lens.rbegin() + 2, s0_lens.rend(), s1_lens.rbegin() + 2, s1_lens.rend()))
            {
                auto l0_it = s0_lens.begin() + s0_lens.size() - 2;
                std::vector<std::size_t> l0_broadcasted_lens(s0_lens.begin(), l0_it);
                auto l1_it = s1_lens.begin() + s1_lens.size() - 2;
                std::vector<std::size_t> l1_broadcasted_lens(s1_lens.begin(), l1_it);
                auto output_lens =
                    compute_broadcasted_lens(l0_broadcasted_lens, l1_broadcasted_lens);
                l0_broadcasted_lens = output_lens;
                l0_broadcasted_lens.insert(l0_broadcasted_lens.end(), l0_it, s0_lens.end());
                l1_broadcasted_lens = output_lens;
                l1_broadcasted_lens.insert(l1_broadcasted_lens.end(), l1_it, s1_lens.end());
                if(s0_lens != l0_broadcasted_lens)
                {
                    ba0 = info.add_instruction(
                        make_op("multibroadcast", {{"out_lens", l0_broadcasted_lens}}), a0);
                }
                if(s1_lens != l1_broadcasted_lens)
                {
                    ba1 = info.add_instruction(
                        make_op("multibroadcast", {{"out_lens", l1_broadcasted_lens}}), a1);
                }
            }

            // parse a_zero_point and b_zero_point values
            if(args.size() > 2)
            {
                ba0 = info.add_instruction(
                    make_op("convert", {{"target_type", migraphx::shape::float_type}}), ba0);

                ba0 = info.add_common_op("sub", ba0, args[2]);
                if(args.size() > 3)
                {
                    ba1 = info.add_instruction(
                        make_op("convert", {{"target_type", migraphx::shape::float_type}}), ba1);
                    ba1 = info.add_common_op("sub", ba1, args[3]);
                }
                dot_res = info.add_instruction(make_op("dot"), ba0, ba1);
                dot_res = info.add_instruction(
                    make_op("convert", {{"target_type", migraphx::shape::int32_type}}), dot_res);
            }
            else
            {
                dot_res = info.add_instruction(make_op(opd.op_name), ba0, ba1);
            }
        }

        // squeeze the appended or prepended dimensions
        int64_t num_axis = dot_res->get_shape().ndim();
        if(is_a_prepended)
        {
            dot_res = info.add_instruction(make_op("squeeze", {{"axes", {num_axis - 2}}}), dot_res);
            --num_axis;
        }
        if(is_b_appended)
        {
            dot_res = info.add_instruction(make_op("squeeze", {{"axes", {num_axis - 1}}}), dot_res);
        }

        return dot_res;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
