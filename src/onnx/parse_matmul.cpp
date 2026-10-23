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
#include <migraphx/common.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/builder/insert.hpp>
#include <algorithm>
#include <iterator>
#include <numeric>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_matmul : op_parser<parse_matmul>
{
    // Four signed correction terms are bounded by 65536*K before the final result.
    static constexpr std::size_t integer_output_signed_limit = 32767;
    // 16384*K stays within int32; a multiple of four also keeps full chunks eligible for CK.
    static constexpr std::size_t integer_dot_chunk_size = 131068;

    struct integer_output_state
    {
        instruction_ref qa;
        instruction_ref qb;
        instruction_ref za_unsigned;
        instruction_ref zb_unsigned;
        bool zero_a;
        bool zero_b;
    };

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

        // obtaining axis parameter for the dequant op-builder
        const auto& lens     = a0->get_shape().lens();
        const auto scale_len = scale_a0->get_shape().lens().at(0);
        const auto rit       = std::find(lens.rbegin(), lens.rend(), scale_len);
        const int axis       = (rit != lens.rend()) ? static_cast<int>(lens.rend() - rit - 1) : 1;

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
        auto res   = op::builder::add("dot", *info.mod, {dq_a0, dq_a1}).at(0);

        // Handle case of the bias after scaling
        if(has_scale_bias)
            res =
                op::builder::insert_common_op(*info.mod, info.mod->end(), "sub", res, scaled_bias);

        return res;
    }

    static instruction_ref integer_operand(const onnx_parser::node_info& info,
                                           instruction_ref input)
    {
        if(input->get_shape().type() != shape::uint8_type)
            return input;
        auto wide =
            info.add_instruction(make_op("convert", {{"target_type", shape::int32_type}}), input);
        auto offset  = info.add_literal(literal{shape{shape::int32_type}, {128}});
        auto shifted = info.add_common_op("sub", wide, offset);
        return info.add_instruction(make_op("convert", {{"target_type", shape::int8_type}}),
                                    shifted);
    }

    static instruction_ref integer_zero_point(const onnx_parser::node_info& info,
                                              const std::vector<instruction_ref>& args,
                                              std::size_t index,
                                              instruction_ref input,
                                              bool lhs)
    {
        instruction_ref zp;
        if(args.size() > index and args[index]->name() != "undefined")
        {
            zp = args[index];
            if(zp->get_shape().type() != input->get_shape().type())
                MIGRAPHX_THROW("MatMulInteger: zero point must be the same type as data");
            if(zp->get_shape().dynamic())
                MIGRAPHX_THROW("MatMulInteger: dynamic zero points not supported");
            if(is_symmetric_zero_point(zp))
                return info.add_literal(literal{shape{shape::int32_type}, {0}});
            const auto& lens = zp->get_shape().lens();
            if(zp->get_shape().elements() == 1)
                zp = info.add_instruction(make_op("reshape", {{"dims", {1}}}), zp);
            else
            {
                auto axis = input->get_shape().ndim() - (lhs ? 2 : 1);
                if(lens.size() == 1 and lens[0] == input->get_shape().lens()[axis])
                {
                    if(lhs)
                        zp = info.add_instruction(make_op("unsqueeze", {{"axes", {1}}}), zp);
                }
                else
                {
                    auto expected                             = input->get_shape().lens();
                    expected[expected.size() - (lhs ? 1 : 2)] = 1;
                    // Batch dimensions may broadcast, but zero points cannot vary along K.
                    if(not can_multibroadcast(lens, expected))
                        MIGRAPHX_THROW(
                            "MatMulInteger: zero point must be scalar or match rows/columns");
                }
            }
            zp = info.add_instruction(make_op("convert", {{"target_type", shape::int32_type}}), zp);
        }
        else
            zp = info.add_literal(literal{shape{shape::int32_type}, {0}});
        if(input->get_shape().type() == shape::uint8_type)
        {
            auto offset = info.add_literal(literal{shape{shape::int32_type}, {128}});
            zp          = info.add_common_op("sub", zp, offset);
        }
        return zp;
    }

    static instruction_ref handle_integer_output_chunk(const onnx_parser::node_info& info,
                                                       const integer_output_state& state,
                                                       std::size_t start,
                                                       std::size_t end)
    {
        auto qa = state.qa;
        auto qb = state.qb;
        if(start != 0 or end != qa->get_shape().lens().back())
        {
            qa = info.add_instruction(
                make_op(
                    "slice",
                    {{"axes", {qa->get_shape().ndim() - 1}}, {"starts", {start}}, {"ends", {end}}}),
                qa);
            qb = info.add_instruction(
                make_op(
                    "slice",
                    {{"axes", {qb->get_shape().ndim() - 2}}, {"starts", {start}}, {"ends", {end}}}),
                qb);
        }

        auto raw_dot = op::builder::add("quant_dot", *info.mod, {qa, qb}).at(0);
        auto result  = info.add_instruction(
            make_op("bit_cast", {{"target_type", shape::uint32_type}}), raw_dot);

        // Unsigned operations make every correction wrap modulo 2^32 by definition.
        if(not state.zero_a)
        {
            auto wb =
                info.add_instruction(make_op("convert", {{"target_type", shape::int32_type}}), qb);
            auto sum_b = info.add_instruction(
                make_op("reduce_sum", {{"axes", {qb->get_shape().ndim() - 2}}}), wb);
            auto sum_b_unsigned = info.add_instruction(
                make_op("bit_cast", {{"target_type", shape::uint32_type}}), sum_b);
            result = info.add_common_op(
                "sub", result, info.add_common_op("mul", state.za_unsigned, sum_b_unsigned));
        }
        if(not state.zero_b)
        {
            auto wa =
                info.add_instruction(make_op("convert", {{"target_type", shape::int32_type}}), qa);
            auto sum_a = info.add_instruction(
                make_op("reduce_sum", {{"axes", {qa->get_shape().ndim() - 1}}}), wa);
            auto sum_a_unsigned = info.add_instruction(
                make_op("bit_cast", {{"target_type", shape::uint32_type}}), sum_a);
            result = info.add_common_op(
                "sub", result, info.add_common_op("mul", sum_a_unsigned, state.zb_unsigned));
        }
        if(not state.zero_a and not state.zero_b)
        {
            auto k = info.add_literal(
                literal{shape{shape::uint32_type}, {static_cast<uint32_t>(end - start)}});
            auto correction = info.add_common_op(
                "mul", info.add_common_op("mul", state.za_unsigned, state.zb_unsigned), k);
            result = info.add_common_op("add", result, correction);
        }
        return result;
    }

    static instruction_ref handle_integer_output_large(const onnx_parser::node_info& info,
                                                       const integer_output_state& state,
                                                       std::size_t k)
    {
        // The raw quant_dot still accumulates to int32, so keep every reduction within this bound.
        const auto number_of_chunks = 1 + (k - 1) / integer_dot_chunk_size;
        std::vector<instruction_ref> chunks;
        chunks.reserve(number_of_chunks);
        std::generate_n(
            std::back_inserter(chunks), number_of_chunks, [&, start = std::size_t{0}]() mutable {
                const auto end = start + std::min(integer_dot_chunk_size, k - start);
                auto result    = handle_integer_output_chunk(info, state, start, end);
                start          = end;
                return result;
            });
        auto result = std::accumulate(
            std::next(chunks.begin()), chunks.end(), chunks.front(), [&](auto x, auto y) {
                return info.add_common_op("add", x, y);
            });
        return info.add_instruction(make_op("bit_cast", {{"target_type", shape::int32_type}}),
                                    result);
    }

    static instruction_ref handle_integer_output(const onnx_parser::node_info& info,
                                                 const std::vector<instruction_ref>& args,
                                                 instruction_ref a,
                                                 instruction_ref b)
    {
        auto za           = integer_zero_point(info, args, 2, a, true);
        auto zb           = integer_zero_point(info, args, 3, b, false);
        auto qa           = integer_operand(info, a);
        auto qb           = integer_operand(info, b);
        const bool zero_a = is_symmetric_zero_point(za);
        const bool zero_b = is_symmetric_zero_point(zb);
        const auto k      = a->get_shape().lens().back();

        if(k != b->get_shape().lens()[b->get_shape().ndim() - 2])
        {
            MIGRAPHX_THROW("MatMulInteger: input inner dimensions must match");
        }

        if(k <= integer_output_signed_limit)
        {
            auto result = op::builder::add("quant_dot", *info.mod, {qa, qb}).at(0);
            if(not zero_a)
            {
                auto wb = info.add_instruction(
                    make_op("convert", {{"target_type", shape::int32_type}}), qb);
                auto sum_b = info.add_instruction(
                    make_op("reduce_sum", {{"axes", {b->get_shape().ndim() - 2}}}), wb);
                result = info.add_common_op("sub", result, info.add_common_op("mul", za, sum_b));
            }
            if(not zero_b)
            {
                auto wa = info.add_instruction(
                    make_op("convert", {{"target_type", shape::int32_type}}), qa);
                auto sum_a = info.add_instruction(
                    make_op("reduce_sum", {{"axes", {a->get_shape().ndim() - 1}}}), wa);
                result = info.add_common_op("sub", result, info.add_common_op("mul", sum_a, zb));
            }
            if(not zero_a and not zero_b)
            {
                auto k_literal =
                    info.add_literal(literal{shape{shape::int32_type}, {static_cast<int32_t>(k)}});
                auto correction =
                    info.add_common_op("mul", info.add_common_op("mul", za, zb), k_literal);
                result = info.add_common_op("add", result, correction);
            }
            return result;
        }

        integer_output_state state{
            qa,
            qb,
            info.add_instruction(make_op("bit_cast", {{"target_type", shape::uint32_type}}), za),
            info.add_instruction(make_op("bit_cast", {{"target_type", shape::uint32_type}}), zb),
            zero_a,
            zero_b};
        return handle_integer_output_large(info, state, k);
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

            dot_res = op::builder::add(opd.op_name, *info.mod, {a0, a1}).at(0);
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

            instruction_ref ba0 =
                is_quant_dot ? a0 : set_bias_arg(op_name, args, a0_zp_index, a0, has_ba0);
            instruction_ref ba1 =
                is_quant_dot ? a1 : set_bias_arg(op_name, args, a1_zp_index, a1, has_ba1);

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

            // Apply the scale to dequantize input to then perform a simple dot
            // after the zero points are applied otherwise get a int32 output from the quantized
            // equivalent. Ensure these are broadcasted accordingly before we perform a dot
            if(is_quant_dot)
            {
                dot_res = handle_integer_output(info, args, a0, a1);
            }
            else if(is_quant_dot_scaled)
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
