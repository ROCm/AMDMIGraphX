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
                                     const instruction_ref& a0,
                                     const instruction_ref& a1,
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

    // Convert to int16 prior to a shift to ensure we preserve accuracy here then
    // convert back to int8
    static instruction_ref add_int8_shift(const onnx_parser::node_info& info,
                                          const instruction_ref& offset_op,
                                          instruction_ref& unshifted_input)
    {
        auto unshifted_input_int16 = info.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int16_type}}),
            unshifted_input);

        auto input_shifted_int16 = info.add_common_op("add", unshifted_input_int16, offset_op);

        return info.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int8_type}}),
            input_shifted_int16);
    }

    static bool is_symmetric_zero_point(instruction_ref zp)
    {
        if(not zp->can_eval())
            return false;

        float check_value = 0;
        if(zp->get_shape().type() == migraphx::shape::uint8_type)
            check_value = 128;

        bool all_zeros = false;
        zp->eval().visit([&](auto z) {
            all_zeros = std::all_of(
                z.begin(), z.end(), [&](auto val) { return float_equal(val, check_value); });
        });
        return all_zeros;
    }

    static instruction_ref set_bias_arg(const std::vector<instruction_ref>& args,
                                        const int index,
                                        const instruction_ref& input,
                                        bool& has_valid_bias)
    {
        has_valid_bias = false;

        if(args.size() > index)
        {
            instruction_ref bias_arg = args[index];
            if(bias_arg->get_shape().type() != input->get_shape().type())
            {
                MIGRAPHX_THROW("PARSE_QUANT_DOT: zero point must be the same type as data");
            }

            // Don't return zero point if it will cause symmetric zero point. No need to bias
            if(is_symmetric_zero_point(bias_arg))
                return input;

            has_valid_bias = true;
            return bias_arg;
        }
        return input;
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

            if(not is_quant_dot and args.size() > 2)
            {
                MIGRAPHX_THROW("PARSE_MATMUL: Bias Args not supported for MatMul");
            }

            bool has_ba0        = false;
            bool has_ba1        = false;
            instruction_ref ba0 = set_bias_arg(args, 2, a0, has_ba0);
            instruction_ref ba1 = set_bias_arg(args, 3, a1, has_ba1);

            // Only INT8 or UINT8 type currently supported
            std::set<migraphx::shape::type_t> supported_types = {migraphx::shape::uint8_type,
                                                                 migraphx::shape::int8_type};
            const auto a0_type                                = a0->get_shape().type();
            const auto a1_type                                = a1->get_shape().type();
            const auto ba0_type                               = ba0->get_shape().type();
            const auto ba1_type                               = ba1->get_shape().type();

            if(is_quant_dot and
               (not contains(supported_types, ba0_type) or
                not contains(supported_types, ba1_type) or not contains(supported_types, a0_type) or
                not contains(supported_types, a1_type)))
            {
                MIGRAPHX_THROW("PARSE_MATMULINTEGER: Unsupported type");
            }

            instruction_ref offset_op;
            if(is_quant_dot and a0_type == migraphx::shape::uint8_type or
               a1_type == migraphx::shape::uint8_type)
            {
                offset_op = info.add_literal(
                    migraphx::literal{migraphx::shape{migraphx::shape::int16_type}, {-128}});
            }

            // always convert uint8 to int8 to avoid rollover
            if(is_quant_dot and a0_type == migraphx::shape::uint8_type)
            {
                a0 = add_int8_shift(info, offset_op, a0);
                if(has_ba0)
                {
                    ba0 = add_int8_shift(info, offset_op, ba0);
                }
                else
                {
                    ba0 = a0;
                }
            }

            if(is_quant_dot and a1_type == migraphx::shape::uint8_type)
            {
                a1 = add_int8_shift(info, offset_op, a1);
                if(has_ba1)
                {
                    ba1 = add_int8_shift(info, offset_op, ba1);
                }
                else
                {
                    ba1 = a1;
                }
            }

            // subtract bias from result after conversion
            if(is_quant_dot and has_ba0)
            {
                ba0 = info.add_common_op("sub", a0, ba0);
            }

            if(is_quant_dot and has_ba1)
            {
                ba1 = info.add_common_op("sub", a1, ba1);
            }

            broadcast_dimensions(info, s0_lens, s1_lens, a0, a1, ba0, ba1);

            dot_res = info.add_instruction(make_op(opd.op_name), ba0, ba1);
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
