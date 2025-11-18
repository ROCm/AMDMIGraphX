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

#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/common.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

template <typename Derived>
struct matmul_base : op_builder<Derived>
{
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

    static instruction_ref set_bias_arg(const std::string& name,
                                        const std::vector<instruction_ref>& args,
                                        const int index,
                                        const instruction_ref& input)
    {
        bool dummy{false};
        return set_bias_arg(name, args, index, input, dummy);
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

    static void broadcast_dimensions(module& m,
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
                ba0 = m.add_instruction(
                    make_op("multibroadcast", {{"out_lens", l0_broadcasted_lens}}), a0);
            }
            if(s1_lens != l1_broadcasted_lens)
            {
                ba1 = m.add_instruction(
                    make_op("multibroadcast", {{"out_lens", l1_broadcasted_lens}}), a1);
            }
        }
    }

    instruction_ref
    insert_impl(module& m, instruction_ref ins, const std::vector<instruction_ref>& args)
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
            a0             = m.add_instruction(make_op("unsqueeze", {{"axes", {0}}}), a0);
        }
        if(s1.ndim() == 1)
        {
            is_b_appended = true;
            a1            = m.add_instruction(make_op("unsqueeze", {{"axes", {1}}}), a1);
        }

        if(s0.dynamic() or s1.dynamic())
        {
            dot_res = static_cast<Derived*>(this)->handle_dynamic(m, a0, a1);
        }
        else
        {
            dot_res = static_cast<Derived*>(this)->handle_static(m, ins, args, a0, a1);
        }

        int64_t num_axis = dot_res->get_shape().ndim();

        if(is_a_prepended)
        {
            dot_res = m.add_instruction(make_op("squeeze", {{"axes", {num_axis - 2}}}), dot_res);
            --num_axis;
        }
        if(is_b_appended)
        {
            dot_res = m.add_instruction(make_op("squeeze", {{"axes", {num_axis - 1}}}), dot_res);
        }

        return dot_res;
    }
};

struct dot : matmul_base<dot>
{
    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args)
    {
        return {insert_impl(m, ins, args)};
    }

    instruction_ref handle_dynamic(module& m, const instruction_ref a0, const instruction_ref a1)
    {
        auto s0_dds = a0->get_shape().to_dynamic().dyn_dims();
        auto s1_dds = a1->get_shape().to_dynamic().dyn_dims();

        if(not std::equal(s0_dds.rbegin() + 2, s0_dds.rend(), s1_dds.rbegin() + 2, s1_dds.rend()))
        {
            auto broadcasted_a0 = m.add_instruction(make_op("broadcast_for_dot"), a0, a1);
            auto broadcasted_a1 = m.add_instruction(make_op("broadcast_for_dot"), a1, a0);
            return m.add_instruction(make_op(name()), broadcasted_a0, broadcasted_a1);
        }
        else
        {
            return m.add_instruction(make_op(name()), a0, a1);
        }
    }

    instruction_ref handle_static(module& m,
                                  instruction_ref,
                                  const std::vector<instruction_ref>& args,
                                  instruction_ref& a0,
                                  instruction_ref& a1)
    {
        if(args.size() > 2)
        {
            MIGRAPHX_THROW(name() + ": Bias Args not supported");
        }

        broadcast_dimensions(m, a0->get_shape().lens(), a1->get_shape().lens(), a0, a1, a0, a1);

        return m.add_instruction(make_op(name()), a0, a1);
    }
};

struct quant_dot : matmul_base<quant_dot>
{
    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args)
    {
        return {insert_impl(m, ins, args)};
    }

    [[noreturn]] instruction_ref
    handle_dynamic(module&, const instruction_ref /*a0*/, const instruction_ref /*a1*/)
    {
        MIGRAPHX_THROW(name() + ": dynamic inputs not supported");
    }

    instruction_ref handle_static(module& m,
                                  instruction_ref ins,
                                  const std::vector<instruction_ref>& args,
                                  instruction_ref& a0,
                                  instruction_ref& a1)
    {
        // Only INT8 or UINT8 type currently supported
        const std::set<migraphx::shape::type_t> supported_types = {migraphx::shape::uint8_type,
                                                                   migraphx::shape::int8_type};
        if((not contains(supported_types, a0->get_shape().type()) or
            not contains(supported_types, a1->get_shape().type())))
        {
            MIGRAPHX_THROW(name() + ": Unsupported type");
        }

        bool has_ba0 = false;
        bool has_ba1 = false;

        const int a0_zp_index = 2;
        const int a1_zp_index = 3;

        instruction_ref ba0 = set_bias_arg(name(), args, a0_zp_index, a0, has_ba0);
        instruction_ref ba1 = set_bias_arg(name(), args, a1_zp_index, a1, has_ba1);

        if((a0->get_shape().type() == migraphx::shape::uint8_type) or
           (a1->get_shape().type() == migraphx::shape::uint8_type))
        {
            auto offset_op = m.add_literal(
                migraphx::literal{migraphx::shape{migraphx::shape::half_type}, {-128}});
            handle_uint8_input(m, ins, has_ba0, offset_op, a0, ba0);
            handle_uint8_input(m, ins, has_ba1, offset_op, a1, ba1);
        }

        broadcast_dimensions(m, a0->get_shape().lens(), a1->get_shape().lens(), a0, a1, ba0, ba1);

        return m.add_instruction(make_op(name()), ba0, ba1);
    }

    private:
    // Convert to half prior to a shift to ensure we preserve accuracy here then
    // convert back to int8
    static instruction_ref add_int8_shift(module& m,
                                          instruction_ref ins,
                                          const instruction_ref& offset_op,
                                          instruction_ref& unshifted_input)
    {
        auto unshifted_input_half = m.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}),
            unshifted_input);

        auto input_shifted_half =
            insert_common_op(m, ins, migraphx::make_op("add"), {unshifted_input_half, offset_op});

        return m.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int8_type}}),
            input_shifted_half);
    }

    static void shift_input_and_bias(module& m,
                                     instruction_ref ins,
                                     const instruction_ref& offset_op,
                                     const bool has_bias,
                                     instruction_ref& input,
                                     instruction_ref& input_bias)
    {
        input = add_int8_shift(m, ins, offset_op, input);
        if(has_bias)
        {
            input_bias = add_int8_shift(m, ins, offset_op, input_bias);
        }
        else
        {
            input_bias = input;
        }
    }

    static void handle_uint8_input(module& m,
                                   instruction_ref ins,
                                   const bool has_bias,
                                   const instruction_ref& offset_op,
                                   instruction_ref& arg,
                                   instruction_ref& bias_arg)
    {
        auto arg_type = arg->get_shape().type();
        // always convert uint8 to int8 to avoid rollover
        if(arg_type == migraphx::shape::uint8_type)
        {
            shift_input_and_bias(m, ins, offset_op, has_bias, arg, bias_arg);
        }

        // subtract bias from result after conversion
        if(has_bias)
        {
            bias_arg = insert_common_op(m, ins, migraphx::make_op("sub"), {arg, bias_arg});
        }
    }
};

struct quant_dot_scaled : matmul_base<quant_dot_scaled>
{
    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args)
    {
        return {insert_impl(m, ins, args)};
    }

    [[noreturn]] instruction_ref
    handle_dynamic(module&, const instruction_ref /*a0*/, const instruction_ref /*a1*/)
    {
        MIGRAPHX_THROW(name() + ": dynamic inputs not supported");
    }

    instruction_ref handle_static(module& m,
                                  instruction_ref ins,
                                  const std::vector<instruction_ref>& args,
                                  instruction_ref& a0,
                                  instruction_ref& a1)
    {
        // Only INT8 or UINT8 type currently supported
        const std::set<migraphx::shape::type_t> supported_types = {migraphx::shape::uint8_type,
                                                                   migraphx::shape::int8_type};
        if((not contains(supported_types, a0->get_shape().type()) or
            not contains(supported_types, a1->get_shape().type())))
        {
            MIGRAPHX_THROW(name() + ": Unsupported type");
        }

        // Handles case with for when scales are present in operator
        instruction_ref scale_a0 = set_scale_arg(m, args, a0, 2);
        instruction_ref scale_a1 = set_scale_arg(m, args, a1, 3);
        if(scale_a0->get_shape().type() != scale_a1->get_shape().type())
        {
            MIGRAPHX_THROW(name() + ": Scales must be the same type");
        }

        const int a0_zp_index = 4;
        const int a1_zp_index = 5;

        instruction_ref ba0 = set_bias_arg(name(), args, a0_zp_index, a0);
        instruction_ref ba1 = set_bias_arg(name(), args, a1_zp_index, a1);

        // handle optional bias arg to the result
        bool has_scale_bias = false;
        auto scaled_index   = 6;
        instruction_ref scaled_bias =
            set_scale_bias(args, scaled_index, scale_a1->get_shape(), a1, has_scale_bias);

        broadcast_dimensions(m, a0->get_shape().lens(), a1->get_shape().lens(), a0, a1, ba0, ba1);

        return handle_scaled_output(
            m, ins, a0, a1, scale_a0, scale_a1, ba0, ba1, scaled_bias, has_scale_bias);
    }

    private:
    static instruction_ref set_scale_arg(module& m,
                                         const std::vector<instruction_ref>& args,
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

        if(scale_shape.scalar())
        {
            scale_arg   = m.add_instruction(make_op("unsqueeze", {{"axes", {0}}}), scale_arg);
        }

        scale_arg = m.add_instruction(make_op("unsqueeze", {{"axes", {0}}}), scale_arg);

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

    static instruction_ref handle_dequantized(module& m,
                                              const instruction_ref& a0,
                                              const instruction_ref& scale_a0,
                                              const instruction_ref& zp_a0,
                                              bool no_zp)
    {
        instruction_ref dequantized_op;

        if(no_zp)
        {
            auto bc_scale_a0 = m.add_instruction(
                make_op("multibroadcast", {{"out_lens", a0->get_shape().lens()}}), scale_a0);
            dequantized_op = m.add_instruction(make_op("dequantizelinear"), a0, bc_scale_a0);
        }
        else
        {
            auto bc_scale_a0 = m.add_instruction(
                make_op("multibroadcast", {{"out_lens", a0->get_shape().lens()}}), scale_a0);

            auto bc_zp_a0 = m.add_instruction(
                make_op("multibroadcast", {{"out_lens", a0->get_shape().lens()}}), zp_a0);

            dequantized_op =
                m.add_instruction(make_op("dequantizelinear"), a0, bc_scale_a0, bc_zp_a0);
        }
        return dequantized_op;
    }

    static instruction_ref handle_scaled_output(module& m,
                                                instruction_ref ins,
                                                const instruction_ref& a0,
                                                const instruction_ref& a1,
                                                const instruction_ref& scale_a0,
                                                const instruction_ref& scale_a1,
                                                const instruction_ref& zp_a0,
                                                const instruction_ref& zp_a1,
                                                const instruction_ref& scaled_bias,
                                                const bool has_scale_bias)
    {

        instruction_ref unsq_zp_a0;
        instruction_ref unsq_zp_a1;

        bool a0_has_no_zp = (a0 == zp_a0);
        bool a1_has_no_zp = (a1 == zp_a1);

        if(not a0_has_no_zp)
        {
            unsq_zp_a0 = m.add_instruction(make_op("unsqueeze", {{"axes", {0}}}), zp_a0);
            if(zp_a0->get_shape().scalar())
            {
                unsq_zp_a0 = m.add_instruction(make_op("unsqueeze", {{"axes", {0}}}), unsq_zp_a0);
            }
        }

        if(not a1_has_no_zp)
        {
            unsq_zp_a1 = m.add_instruction(make_op("unsqueeze", {{"axes", {0}}}), zp_a1);
            if(zp_a1->get_shape().scalar())
            {
                unsq_zp_a1 = m.add_instruction(make_op("unsqueeze", {{"axes", {0}}}), unsq_zp_a1);
            }
        }

        auto dq_a0 = handle_dequantized(m, a0, scale_a0, unsq_zp_a0, a0_has_no_zp);
        auto dq_a1 = handle_dequantized(m, a1, scale_a1, unsq_zp_a1, a1_has_no_zp);
        auto res   = m.add_instruction(make_op("dot"), dq_a0, dq_a1);

        // Handle case of the bias after scaling
        if(has_scale_bias)
            res = insert_common_op(m, ins, migraphx::make_op("sub"), {res, scaled_bias});

        return res;
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
