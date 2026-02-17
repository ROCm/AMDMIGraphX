/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <migraphx/op/builder/insert.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_matmul : op_parser<parse_matmul>
{
    std::vector<op_desc> operators() const
    {
        return {{"MatMul", "dot"},
                {"MatMulInteger", "quant_dot"},
                {"MatMulIntegerToFloat", "quant_dot_scaled"}};
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

    static instruction_ref set_scale_arg(const std::vector<instruction_ref>& args,
                                         const instruction_ref& mat_input,
                                         const int index)
    {
        instruction_ref scale_arg                            = args[index];
        std::set<migraphx::shape::type_t> supported_dq_types = {migraphx::shape::float_type,
                                                                migraphx::shape::half_type};

        auto scale_shape = scale_arg->get_shape();

        if(not(contains(supported_dq_types, scale_shape.type())))
        {
            MIGRAPHX_THROW("PARSE_QUANT_DOT_SCALED: Scales must be float or half_type");
        }

        if(scale_shape.lens().at(0) != *(mat_input->get_shape().lens().rbegin()) and
           not scale_shape.scalar())
        {
            MIGRAPHX_THROW("PARSE_QUANT_DOT_SCALED: Scale must have same dim as matrix column");
        }

        if(scale_shape.lens().size() > 1 and not scale_shape.scalar())
        {
            MIGRAPHX_THROW("PARSE_QUANT_DOT_SCALED: Scales shape must be scalar or 1-D tensor");
        }

        return scale_arg;
    }

    static instruction_ref set_scale_bias(const std::vector<instruction_ref>& args,
                                          const int index,
                                          const migraphx::shape& scale_arg_shape,
                                          const instruction_ref& compare_arg,
                                          bool& has_valid_scale_bias)
    {
        has_valid_scale_bias = false;

        if(args.size() > index)
        {
            instruction_ref scale_bias_arg                       = args[index];
            std::set<migraphx::shape::type_t> supported_dq_types = {migraphx::shape::float_type,
                                                                    migraphx::shape::half_type};

            if(not(contains(supported_dq_types, scale_bias_arg->get_shape().type())))
            {
                MIGRAPHX_THROW("PARSE_QUANT_DOT_SCALED: Bias must be float or half_type");
            }

            if(scale_bias_arg->get_shape().type() != scale_arg_shape.type())
            {
                MIGRAPHX_THROW("PARSE_QUANT_DOT_SCALED: Bias must be the same type as scales");
            }

            if(scale_bias_arg->get_shape().lens().at(0) !=
               *(compare_arg->get_shape().lens().rbegin()))
            {
                MIGRAPHX_THROW("PARSE_QUANT_DOT_SCALED: Bias have same dim as matrix B column");
            }

            has_valid_scale_bias = true;
            return scale_bias_arg;
        }
        return compare_arg;
    }

    static instruction_ref set_bias_arg(const std::string& name,
                                        const std::vector<instruction_ref>& args,
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
                MIGRAPHX_THROW(name + ": zero point must be the same type as data");
            }

            // Don't return zero point if it will cause symmetric zero point. No need to bias
            if(is_symmetric_zero_point(bias_arg))
                return input;

            has_valid_bias = true;
            return bias_arg;
        }
        return input;
    }

    static instruction_ref handle_dequantized(const onnx_parser::node_info& info,
                                              const instruction_ref& a0,
                                              const instruction_ref& scale_a0,
                                              const instruction_ref& zp_a0,
                                              bool no_zp)
    {
        instruction_ref dequantized_op;

        std::vector<instruction_ref> dq_args{a0, scale_a0};
        if(not no_zp)
        {
            dq_args.push_back(zp_a0);
        }

        const auto& lens = a0->get_shape().lens();
        const auto scale_len = scale_a0->get_shape().lens().at(0);
        const auto rit = std::find(lens.rbegin(), lens.rend(), scale_len);
        const int axis = (rit != lens.rend()) ? static_cast<int>(lens.rend() - rit - 1) : 1;

        dequantized_op =
            op::builder::add("dequantizelinear", *info.mod, dq_args, {{"axis", axis}}).at(0);

        return dequantized_op;
    }

    static instruction_ref handle_scaled_output(const onnx_parser::node_info& info,
                                                const instruction_ref& a0,
                                                const instruction_ref& a1,
                                                const instruction_ref& scale_a0,
                                                const instruction_ref& scale_a1,
                                                const instruction_ref& zp_a0,
                                                const instruction_ref& zp_a1,
                                                const instruction_ref& scaled_bias,
                                                const bool has_scale_bias)
    {
        auto dq_a0 = handle_dequantized(info, a0, scale_a0, zp_a0, (a0 == zp_a0));
        auto dq_a1 = handle_dequantized(info, a1, scale_a1, zp_a1, (a1 == zp_a1));
        auto res   = info.add_instruction(make_op("dot"), dq_a0, dq_a1);

        // Handle case of the bias after scaling
        if(has_scale_bias)
            res = info.add_common_op("sub", res, scaled_bias);

        return res;
    }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        std::string op_name{opd.op_name};
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
            a0 = op::builder::add("unsqueeze", *info.mod, {args[0]}, {{"axes", {0}}}).at(0);
        }
        if(s1.ndim() == 1)
        {
            is_b_appended = true;
            a1 = op::builder::add("unsqueeze", *info.mod, {args[1]}, {{"axes", {1}}}).at(0);
        }

        auto is_quant_dot        = opd.op_name == "quant_dot";
        auto is_quant_dot_scaled = opd.op_name == "quant_dot_scaled";
        auto is_dot              = opd.op_name == "dot";

        if(s0.dynamic() or s1.dynamic())
        {
            if(is_quant_dot or is_quant_dot_scaled)
            {
                MIGRAPHX_THROW(op_name + ": dynamic inputs not supported");
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
            if(is_dot and args.size() > 2)
            {
                MIGRAPHX_THROW(op_name + ": Bias Args not supported");
            }

            bool has_ba0        = false;
            bool has_ba1        = false;
            bool has_scale_bias = false;

            int a0_zp_index = 2;
            int a1_zp_index = 3;

            instruction_ref scale_a0;
            instruction_ref scale_a1;
            // Handles case with for when scales are present in operator
            if(is_quant_dot_scaled)
            {
                a0_zp_index = 4;
                a1_zp_index = 5;
                scale_a0    = set_scale_arg(args, a0, 2);
                scale_a1    = set_scale_arg(args, a1, 3);
                if(scale_a0->get_shape().type() != scale_a1->get_shape().type())
                {
                    MIGRAPHX_THROW(op_name + ": Scales must be the same type");
                }
            }

            instruction_ref ba0 = set_bias_arg(op_name, args, a0_zp_index, a0, has_ba0);
            instruction_ref ba1 = set_bias_arg(op_name, args, a1_zp_index, a1, has_ba1);

            // handle optional bias arg to the result
            instruction_ref scaled_bias;
            if(is_quant_dot_scaled)
            {
                auto scaled_index = 6;
                scaled_bias =
                    set_scale_bias(args, scaled_index, scale_a1->get_shape(), a1, has_scale_bias);
            }

            // Only INT8 or UINT8 type currently supported
            std::set<migraphx::shape::type_t> supported_types = {migraphx::shape::uint8_type,
                                                                 migraphx::shape::int8_type};
            const auto a0_type                                = a0->get_shape().type();
            const auto a1_type                                = a1->get_shape().type();

            if((not is_dot) and
               (not contains(supported_types, a0_type) or not contains(supported_types, a1_type)))
            {
                MIGRAPHX_THROW(op_name + ": Unsupported type");
            }

            if((is_quant_dot and ((a0_type == migraphx::shape::uint8_type) or
                                  (a1_type == migraphx::shape::uint8_type))))
            {
                auto unpack2 = [](auto&& v) { return std::make_pair(v[0], v[1]); };

                std::tie(a0, ba0) = unpack2(op::builder::add("bias_uint8", *info.mod, {a0, ba0}, {{"has_bias", has_ba0}}));
                std::tie(a1, ba1) = unpack2(op::builder::add("bias_uint8", *info.mod, {a1, ba1}, {{"has_bias", has_ba1}}));
            }

            // Apply the scale to dequantize input to then perform a simple dot
            // after the zero points are applied otherwise get a int32 output from the quantized
            // equivalent. Ensure these are broadcasted accordingly before we perform a dot
            if(is_quant_dot_scaled)
            {
                dot_res = handle_scaled_output(
                    info, a0, a1, scale_a0, scale_a1, ba0, ba1, scaled_bias, has_scale_bias);
            }
            else
            {
                dot_res = op::builder::add(opd.op_name, *info.mod, {a0, a1, ba0, ba1}).at(0);
            }
        }

        // squeeze the appended or prepended dimensions
        int64_t num_axis = dot_res->get_shape().ndim();
        if(is_a_prepended)
        {
            dot_res =
                op::builder::add("squeeze", *info.mod, {dot_res}, {{"axes", {num_axis - 2}}}).at(0);
            --num_axis;
        }
        if(is_b_appended)
        {
            dot_res =
                op::builder::add("squeeze", *info.mod, {dot_res}, {{"axes", {num_axis - 1}}}).at(0);
        }

        return dot_res;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
