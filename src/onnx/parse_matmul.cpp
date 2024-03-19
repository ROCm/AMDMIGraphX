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

    static void broadcast_dimensions(const onnx_parser::node_info& info,
                                     const std::vector<size_t>& s0_lens,
                                     const std::vector<size_t>& s1_lens,
                                     instruction_ref& a0,
                                     instruction_ref& a1,
                                     instruction_ref& ba0,
                                     instruction_ref& ba1)
    {
        // try broadcasting if dimensions other than last two do not match
        if(not std::equal(
               s0_lens.rbegin() + 2, s0_lens.rend(), s1_lens.rbegin() + 2, s1_lens.rend()))
        {
            auto l0_it = s0_lens.begin() + s0_lens.size() - 2;
            std::vector<std::size_t> l0_broadcasted_lens(s0_lens.begin(), l0_it);
            auto l1_it = s1_lens.begin() + s1_lens.size() - 2;
            std::vector<std::size_t> l1_broadcasted_lens(s1_lens.begin(), l1_it);
            auto output_lens = compute_broadcasted_lens(l0_broadcasted_lens, l1_broadcasted_lens);
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
    }

    static void
    add_int8_shift(const onnx_parser::node_info& info, instruction_ref& ba0, instruction_ref& ba1)
    {
        const auto ba0_type = ba0->get_shape().type();
        const auto ba1_type = ba1->get_shape().type();

        // determine if we need to add a shift to ensure items get shifted to the same int8
        // type if we see uint8/int8 mismatches
        if(ba0_type == ba1_type)
            return;

        instruction_ref unshifted_input;
        instruction_ref* shifted_output = nullptr;
        if(ba0_type == migraphx::shape::int8_type)
        {
            unshifted_input = ba1;
            shifted_output  = &ba1;
        }
        if(ba1_type == migraphx::shape::int8_type)
        {
            unshifted_input = ba0;
            shifted_output  = &ba0;
        }

        // Convert to fp16 prior to a shift to ensure we preserve accuracy here then
        // convert back to int8
        auto int8_shift = info.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int16_type}, {-128}});

        auto unshifted_input_int16 = info.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int16_type}}),
            unshifted_input);

        auto input_shifted_int16 = info.add_common_op("add", unshifted_input_int16, int8_shift);

        auto shifted_input = info.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int8_type}}),
            input_shifted_int16);

        *shifted_output = shifted_input;
    }

    static void set_bias_arg(const onnx_parser::node_info& info,
                             const std::vector<instruction_ref>& args,
                             const int index,
                             const instruction_ref& input,
                             instruction_ref& bias_arg)
    {
        if(args.size() > index)
        {
            if(args[index]->get_shape().type() != input->get_shape().type())
            {
                MIGRAPHX_THROW("PARSE_QUANT_DOT: zero point must be the same type as data");
            }

            bias_arg = args[index];
            bias_arg = info.add_common_op("sub", input, bias_arg);
        }
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
                auto broadcasted_a0 = info.add_instruction(make_op("broadcast_for_dot"), a0, a1);
                auto broadcasted_a1 = info.add_instruction(make_op("broadcast_for_dot"), a1, a0);
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
            if(is_quant_dot)
            {
                set_bias_arg(info, args, 2, a0, ba0);
                set_bias_arg(info, args, 3, a1, ba1);

                add_int8_shift(info, ba0, ba1);
                broadcast_dimensions(info, s0_lens, s1_lens, a0, a1, ba0, ba1);

                dot_res = info.add_instruction(make_op("quant_dot"), ba0, ba1);
                dot_res = info.add_instruction(
                    make_op("convert", {{"target_type", migraphx::shape::int32_type}}), dot_res);
            }
            else
            {
                broadcast_dimensions(info, s0_lens, s1_lens, a0, a1, ba0, ba1);
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
