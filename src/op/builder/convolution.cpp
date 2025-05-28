/* The MIT License (MIT)
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

#include <migraphx/common.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/op/builder/insert.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

template <class Derived>
struct convolution_base : op_builder<Derived>
{
    std::string auto_pad = "NOTSET";
    std::vector<int64_t> paddings;
    std::vector<std::size_t> strides;
    std::vector<std::size_t> dilations;
    int group                   = 1;
    padding_mode_t padding_mode = padding_mode_t::default_;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.auto_pad, "auto_pad"),
                    f(self.paddings, "paddings"),
                    f(self.strides, "strides"),
                    f(self.dilations, "dilations"),
                    f(self.group, "group"),
                    f(self.padding_mode, "padding_mode"));
    }

    void validate_or_init_attributes(size_t kdims, const instruction_ref x, const instruction_ref w)
    {

        if(strides.empty())
        {
            strides.resize(kdims);
            std::fill_n(strides.begin(), kdims, 1);
        }
        else if(strides.size() != kdims)
        {
            MIGRAPHX_THROW("Inconsistent strides size, is: " + std::to_string(strides.size()) +
                           ", should be: " + std::to_string(kdims));
        }

        if(dilations.empty())
        {
            dilations.resize(kdims);
            std::fill_n(dilations.begin(), kdims, 1);
        }
        else if(dilations.size() != kdims)
        {
            MIGRAPHX_THROW("Inconsistent dilations size, is: " + std::to_string(dilations.size()) +
                           ", should be: " + std::to_string(kdims));
        }

        if(paddings.empty())
        {
            paddings.resize(kdims);
            std::fill_n(paddings.begin(), kdims, 0);
        }
        else if(paddings.size() != kdims and paddings.size() != 2 * kdims)
        {
            MIGRAPHX_THROW("Inconsistent paddings size, is: " + std::to_string(paddings.size()) +
                           ", should be: " + std::to_string(kdims) +
                           " or: " + std::to_string(2 * kdims));
        }

        if(contains(auto_pad, "SAME"))
        {
            if(is_dynamic(x->get_shape()) or is_dynamic(w->get_shape()))
            {
                // must calculate "same" padding with input shape data
                padding_mode = contains(auto_pad, "SAME_UPPER") ? op::padding_mode_t::same_upper
                                                                : op::padding_mode_t::same_lower;
            }
            else
            {
                // kernel shape will be fixed, so max_lens() == min_len() for kernel lengths
                auto weight_lens = w->get_shape().max_lens();
                std::vector<std::size_t> k_lens(weight_lens.begin() + 2, weight_lens.end());
                calc_auto_padding(
                    auto_pad, strides, k_lens, dilations, x->get_shape().max_lens(), paddings);
            }
        }
    }

    bool is_dynamic(const shape& s) const
    {
        return s.dynamic() and
               std::any_of(s.dyn_dims().begin() + 2, s.dyn_dims().end(), [](const auto& dyn_dim) {
                   return not dyn_dim.is_fixed();
               });
    }

    // TODO Move this out into a util file
    void calc_auto_padding(std::string auto_pad,
                           const std::vector<std::size_t>& strides,
                           const std::vector<std::size_t>& k_lens,
                           const std::vector<std::size_t>& dilation,
                           const std::vector<std::size_t>& in_lens,
                           std::vector<int64_t>& paddings) const
    {
        size_t kdims = in_lens.size() - 2;
        assert(k_lens.size() == kdims and dilation.size() == kdims);

        auto_pad = to_upper(auto_pad);
        if(contains(auto_pad, "SAME"))
        {
            bool is_same_upper = contains(auto_pad, "SAME_UPPER");
            paddings.resize(2 * kdims);

            for(size_t i = 0; i < paddings.size() / 2; i++)
            {
                calculate_padding(
                    i, paddings, in_lens[i + 2], strides[i], dilation[i], k_lens[i], is_same_upper);
            }
        }
    }

    // TODO Move this out into a util file
    void calculate_padding(int64_t idx,
                           std::vector<int64_t>& pads,
                           int64_t input_dim,
                           int64_t stride,
                           int64_t dilation,
                           int64_t weight_dim,
                           bool is_same_upper) const
    {
        int64_t output_dim     = (input_dim + stride - 1) / stride; // round up result
        int64_t new_weight_dim = weight_dim + (weight_dim - 1) * (dilation - 1);
        int64_t pad            = std::max(static_cast<int64_t>(0),
                               (output_dim - 1) * stride + new_weight_dim - input_dim);
        auto pad_ndims         = pads.size() / 2;

        if(is_same_upper)
        {
            pads[idx]             = pad / 2;
            pads[idx + pad_ndims] = pad - pad / 2;
        }
        else
        {
            pads[idx + pad_ndims] = pad / 2;
            pads[idx]             = pad - pad / 2;
        }
    }

    operation make_conv_op(const std::string& name) const
    {
        return make_op(name,
                       {{"stride", strides},
                        {"dilation", dilations},
                        {"padding", paddings},
                        {"group", group},
                        {"padding_mode", padding_mode}});
    }
};

struct convolution : convolution_base<convolution>
{
    static std::string name() { return "convolution"; }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args)
    {
        auto x       = args[0];
        auto weights = args[1];
        auto in_lens = x->get_shape().max_lens();
        assert(in_lens.size() > 2);
        auto kdims = in_lens.size() - 2;

        validate_or_init_attributes(kdims, x, weights);

        auto conv = m.insert_instruction(ins, make_conv_op("convolution"), x, weights);
        return {add_bias(m, ins, args, conv, 1)};
    }

    // TODO Move this out into util file
    instruction_ref add_bias(module& m,
                             instruction_ref ins,
                             const std::vector<instruction_ref>& args,
                             instruction_ref curr_ins,
                             uint64_t axis) const
    {
        if(args.size() == 3)
        {
            instruction_ref bias_bcast;
            // if curr_ins has a dynamic output shape use 2 input broadcast
            if(curr_ins->get_shape().dynamic())
            {
                bias_bcast = m.insert_instruction(
                    ins, make_op("broadcast", {{"axis", axis}}), args[2], curr_ins);
            }
            else
            {
                bias_bcast = m.insert_instruction(
                    ins,
                    make_op("broadcast",
                            {{"axis", axis}, {"out_lens", curr_ins->get_shape().lens()}}),
                    args[2]);
            }
            return m.insert_instruction(ins, make_op("add"), curr_ins, bias_bcast);
        }
        return curr_ins;
    }
};

struct convolution_integer : convolution_base<convolution_integer>
{
    static std::string name() { return "convolution_integer"; }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args)
    {
        auto x       = args[0];
        auto weights = args[1];
        auto in_lens = x->get_shape().max_lens();
        assert(in_lens.size() > 2);
        auto kdims = in_lens.size() - 2;

        validate_or_init_attributes(kdims, x, weights);

        auto op   = make_conv_op("quant_convolution");
        auto x_zp = get_zero_point(m, ins, x, 2, args);
        auto w_zp = get_zero_point(m, ins, weights, 3, args);
        handle_quant_inputs(m, ins, x, weights, x_zp, w_zp);
        auto conv = m.insert_instruction(ins, op, x, weights);
        return {handle_quant_bias(m, ins, op, conv, x, weights, x_zp, w_zp)};
    }

    // Convert to half prior to a shift to ensure we preserve accuracy here then
    // convert back to int8
    instruction_ref add_int8_shift(module& m,
                                   instruction_ref ins,
                                   const instruction_ref& offset_op,
                                   instruction_ref& unshifted_input) const
    {
        auto unshifted_input_half = m.insert_instruction(
            ins,
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}),
            unshifted_input);

        auto input_shifted_half = insert_common_op(m, ins, "add", unshifted_input_half, offset_op);

        return m.insert_instruction(
            ins,
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int8_type}}),
            input_shifted_half);
    }

    void shift_input_and_bias(module& m,
                              instruction_ref ins,
                              const instruction_ref& offset_op,
                              const bool has_bias,
                              instruction_ref& input,
                              instruction_ref& input_bias) const
    {
        input = add_int8_shift(m, ins, offset_op, input);
        if(has_bias)
        {
            input_bias = add_int8_shift(m, ins, offset_op, input_bias);
        }
    }

    float get_symmetric_value(const instruction_ref& input) const
    {
        float symmetric_value = 0;
        // adjust symmetric zero point value for uint8 types
        if(input->get_shape().type() == migraphx::shape::uint8_type)
        {
            symmetric_value = 128;
        }
        return symmetric_value;
    }

    instruction_ref
    gen_symmetric_literal(module& m, instruction_ref ins, const instruction_ref& input) const
    {
        float symmetric_value = get_symmetric_value(input);
        return m.insert_literal(ins, {{input->get_shape().type(), {1}, {0}}, {symmetric_value}});
    }

    instruction_ref get_zero_point(module& m,
                                   instruction_ref ins,
                                   const instruction_ref& input,
                                   int index,
                                   const std::vector<instruction_ref>& args) const
    {
        instruction_ref ret = input;
        if(args.size() > index)
        {
            // Check for type mismatch on parse
            if(input->get_shape().type() != args[index]->get_shape().type())
                MIGRAPHX_THROW("PARSE:Conv Data and Data Zero Point must have same type");

            ret = args[index];
            if(is_symmetric_zero_point(ret))
            {
                ret = gen_symmetric_literal(m, ins, ret);
            }
        }
        else
        {
            ret = gen_symmetric_literal(m, ins, ret);
        }

        return ret;
    }

    bool is_symmetric_zero_point(instruction_ref zp) const
    {
        if(not zp->can_eval())
            return false;

        float symmetric_value = get_symmetric_value(zp);

        bool all_zeros = false;
        zp->eval().visit([&](auto z) {
            all_zeros = std::all_of(
                z.begin(), z.end(), [&](auto val) { return float_equal(val, symmetric_value); });
        });
        return all_zeros;
    }

    auto qparam_broadcast_op(instruction_ref qparam,
                             std::vector<std::size_t> lens,
                             std::size_t axis) const
    {
        if(qparam->get_shape().scalar())
        {
            return migraphx::make_op("multibroadcast", {{"out_lens", lens}});
        }
        else
        {
            return migraphx::make_op("broadcast", {{"out_lens", lens}, {"axis", axis}});
        }
    }

    instruction_ref handle_quant_bias(module& m,
                                      instruction_ref ins,
                                      const operation& op,
                                      const instruction_ref& input,
                                      const instruction_ref& x,
                                      const instruction_ref& weights,
                                      const instruction_ref& x_zp,
                                      const instruction_ref& w_zp) const
    {
        instruction_ref ret = input;
        if(not is_symmetric_zero_point(x_zp))
        {
            auto out_zp_1 = insert_common_op(m, ins, op.name(), x_zp, weights);
            ret           = insert_common_op(m, ins, "sub", ret, out_zp_1);
        }

        if(not is_symmetric_zero_point(w_zp))
        {
            auto out_zp_2 = insert_common_op(m, ins, op.name(), x, w_zp);
            ret           = insert_common_op(m, ins, "sub", ret, out_zp_2);
        }

        if(not(is_symmetric_zero_point(x_zp)) and not(is_symmetric_zero_point(w_zp)))
        {
            auto x_zp_bc = m.insert_instruction(
                ins, qparam_broadcast_op(x_zp, x->get_shape().lens(), 0), x_zp);
            auto w_zp_bc = m.insert_instruction(
                ins, qparam_broadcast_op(w_zp, weights->get_shape().lens(), 0), w_zp);

            auto out_zp_3 = m.insert_instruction(ins, op, x_zp_bc, w_zp_bc);

            ret = insert_common_op(m, ins, "add", ret, out_zp_3);
        }
        return ret;
    }

    void handle_quant_inputs(module& m,
                             instruction_ref ins,
                             instruction_ref& input,
                             instruction_ref& weights,
                             instruction_ref& input_zp,
                             instruction_ref& weight_zp) const
    {
        auto input_type  = input->get_shape().type();
        auto weight_type = weights->get_shape().type();

        // Handle uint8 bias and input shifts
        instruction_ref offset_op;
        if(((input_type == migraphx::shape::uint8_type) or
            (weight_type == migraphx::shape::uint8_type)))
        {
            offset_op = m.insert_literal(ins, {{migraphx::shape::half_type}, {-128}});
        }

        if(input_type == migraphx::shape::uint8_type)
        {
            shift_input_and_bias(
                m, ins, offset_op, (not is_symmetric_zero_point(input_zp)), input, input_zp);
        }

        if(weight_type == migraphx::shape::uint8_type)
        {
            shift_input_and_bias(
                m, ins, offset_op, (not is_symmetric_zero_point(weight_zp)), weights, weight_zp);
        }
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
