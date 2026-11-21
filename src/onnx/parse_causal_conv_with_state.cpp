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
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/op/builder/insert.hpp>
#include <limits>
#include <numeric>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

// ONNXRuntime contrib op, reference implementation:
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/contrib_ops/cpu/bert/causal_conv_with_state.cc
//
// The output is a depthwise 1D convolution over the carry state followed by the input, so an
// output position only reads itself and earlier positions. present_state is the tail of that
// sequence, one slot per trailing position when state_window > 0.
struct causal_conv_parameters
{
    // Attributes
    std::string activation = "none";
    int64_t dilation       = 1;
    bool channels_last     = false;
    int64_t state_window   = 0;

    // Derived from the inputs
    std::vector<std::size_t> input_lens;
    std::size_t causal_axis = 2; // sequence axis of the input, output and each carry state
    std::size_t channels    = 0;
    std::vector<std::size_t> channel_lens; // channel axes of the input, several with channels_last
    int64_t state_length = 0;              // positions carried over: (kernel_size - 1) * dilation
    std::vector<std::size_t> slot_lens;    // one carry state: the input with state_length positions
};

struct parse_causal_conv_with_state : op_parser<parse_causal_conv_with_state>
{
    std::vector<op_desc> operators() const { return {{"CausalConvWithState"}}; }

    static void parse_attributes(const onnx_parser& parser,
                                 const onnx_parser::node_info& info,
                                 causal_conv_parameters& params)
    {
        if(contains(info.attributes, "activation"))
            params.activation = info.attributes.at("activation").s();
        if(not contains({"none", "silu", "swish"}, params.activation))
            MIGRAPHX_THROW("CausalConvWithState: activation must be none, silu or swish, got " +
                           params.activation);
        // No ONNXRuntime backend implements the 2D and 3D variants either
        if(contains(info.attributes, "ndim") and
           parser.parse_value(info.attributes.at("ndim")).at<int64_t>() != 1)
            MIGRAPHX_THROW("CausalConvWithState: only ndim=1 is supported");
        if(contains(info.attributes, "dilation"))
            params.dilation = parser.parse_value(info.attributes.at("dilation")).at<int64_t>();
        if(params.dilation < 1)
            MIGRAPHX_THROW("CausalConvWithState: dilation must be >= 1, got " +
                           std::to_string(params.dilation));
        if(contains(info.attributes, "channels_last"))
        {
            auto channels_last =
                parser.parse_value(info.attributes.at("channels_last")).at<int64_t>();
            if(channels_last != 0 and channels_last != 1)
                MIGRAPHX_THROW("CausalConvWithState: channels_last must be 0 or 1, got " +
                               std::to_string(channels_last));
            params.channels_last = channels_last == 1;
        }
        if(contains(info.attributes, "state_window"))
            params.state_window =
                parser.parse_value(info.attributes.at("state_window")).at<int64_t>();
        if(params.state_window < 0 or params.state_window > 8)
            MIGRAPHX_THROW("CausalConvWithState: state_window must be in [0, 8], got " +
                           std::to_string(params.state_window));
    }

    // Optional inputs left empty in the model arrive as undefined instructions
    static bool has_arg(const std::vector<instruction_ref>& args, std::size_t index)
    {
        return args.size() > index and not args.at(index)->is_undefined();
    }

    static void parse_inputs(const std::vector<instruction_ref>& args,
                             causal_conv_parameters& params)
    {
        if(args.size() < 2 or args.size() > 4)
            MIGRAPHX_THROW("CausalConvWithState: expected 2 to 4 inputs, got " +
                           std::to_string(args.size()));
        params.input_lens = args[0]->get_shape().max_lens();
        auto weight_lens  = args[1]->get_shape().max_lens();
        // channels_last: (batch, seq_len, d_1, ..., d_n); otherwise (batch, channels, seq_len)
        params.causal_axis = params.channels_last ? 1 : 2;
        if(params.channels_last ? params.input_lens.size() < 3 : params.input_lens.size() != 3)
            MIGRAPHX_THROW("CausalConvWithState: input must be (batch, channels, seq_len), or "
                           "(batch, seq_len, channels...) with channels_last");
        if(weight_lens.size() != 3 or weight_lens[1] != 1 or weight_lens[2] == 0)
            MIGRAPHX_THROW("CausalConvWithState: weight must be (channels, 1, kernel_size)");
        params.channels     = weight_lens[0];
        params.channel_lens = {params.input_lens[1]};
        if(params.channels_last)
            params.channel_lens.assign(params.input_lens.begin() + 2, params.input_lens.end());
        if(std::accumulate(params.channel_lens.begin(),
                           params.channel_lens.end(),
                           std::size_t{1},
                           std::multiplies<>{}) != params.channels)
            MIGRAPHX_THROW("CausalConvWithState: input channels do not match the weight");
        params.state_length                  = (weight_lens[2] - 1) * params.dilation;
        params.slot_lens                     = params.input_lens;
        params.slot_lens[params.causal_axis] = params.state_length;

        if(has_arg(args, 2) and
           args[2]->get_shape().max_lens() != std::vector<std::size_t>{params.channels})
            MIGRAPHX_THROW("CausalConvWithState: bias must be (channels)");
        auto state_lens = params.slot_lens;
        if(params.state_window > 0)
            state_lens.insert(state_lens.begin(), params.state_window);
        if(has_arg(args, 3) and args[3]->get_shape().max_lens() != state_lens)
            MIGRAPHX_THROW("CausalConvWithState: past_state must be (" +
                           to_string_range(state_lens) + "), got (" +
                           to_string_range(args[3]->get_shape().max_lens()) + ")");
    }

    // The carry state followed by the input along the sequence axis
    static instruction_ref insert_sequence(const onnx_parser::node_info& info,
                                           const std::vector<instruction_ref>& args,
                                           const causal_conv_parameters& params)
    {
        if(not has_arg(args, 3))
        {
            // Without a carry state the positions before the input are zero
            std::vector<int64_t> pads(2 * params.input_lens.size(), 0);
            pads[params.causal_axis] = params.state_length;
            return info.add_instruction(make_op("pad", {{"pads", pads}}), args[0]);
        }
        auto past = args[3];
        if(params.state_window > 0)
        {
            // Only the slot after the last position carries forward
            past = info.add_instruction(make_op("slice",
                                                {{"axes", {0}},
                                                 {"starts", {params.state_window - 1}},
                                                 {"ends", {params.state_window}}}),
                                        past);
            past = info.add_instruction(make_op("squeeze", {{"axes", {0}}}), past);
        }
        return info.add_instruction(
            make_op("concat", {{"axis", params.causal_axis}}), past, args[0]);
    }

    static instruction_ref insert_output(const onnx_parser::node_info& info,
                                         const std::vector<instruction_ref>& args,
                                         const causal_conv_parameters& params,
                                         instruction_ref seq)
    {
        auto x = seq;
        if(params.channels_last)
        {
            // The convolution needs a single channel axis ahead of the sequence axis
            if(params.input_lens.size() > 3)
                x = info.add_instruction(make_op("reshape", {{"dims", {0, 0, -1}}}), x);
            x = info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1}}}), x);
        }
        std::vector<instruction_ref> conv_args = {x, args[1]};
        if(has_arg(args, 2))
            conv_args.push_back(args[2]);
        auto output =
            op::builder::add("convolution",
                             *info.mod,
                             conv_args,
                             {{"group", params.channels}, {"dilations", {params.dilation}}})
                .at(0);
        if(params.activation != "none")
        {
            auto sigmoid = info.add_instruction(make_op("sigmoid"), output);
            output       = info.add_instruction(make_op("mul"), output, sigmoid);
        }
        if(params.channels_last)
        {
            output =
                info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1}}}), output);
            if(params.input_lens.size() > 3)
            {
                std::vector<int64_t> dims = {0, 0};
                dims.insert(dims.end(), params.channel_lens.begin(), params.channel_lens.end());
                output = info.add_instruction(make_op("reshape", {{"dims", dims}}), output);
            }
        }
        return output;
    }

    // The carry state after the position `back` places before the end of the sequence: its last
    // state_length positions once the trailing `back` positions are dropped. Counted from the end
    // so a dynamic sequence length works.
    static instruction_ref slice_state(const onnx_parser::node_info& info,
                                       const causal_conv_parameters& params,
                                       instruction_ref seq,
                                       int64_t back)
    {
        int64_t end = -back;
        if(back == 0 and params.state_length > 0)
            end = std::numeric_limits<int64_t>::max();
        return info.add_instruction(make_op("slice",
                                            {{"axes", {params.causal_axis}},
                                             {"starts", {-(params.state_length + back)}},
                                             {"ends", {end}}}),
                                    seq);
    }

    static instruction_ref insert_present_state(const onnx_parser::node_info& info,
                                                const std::vector<instruction_ref>& args,
                                                const causal_conv_parameters& params,
                                                instruction_ref seq)
    {
        if(params.state_window == 0)
            return slice_state(info, params, seq, 0);
        // Slot j is the state after input position seq_len - state_window + j; slots before the
        // first position hold nothing from this call and are zero.
        const auto& input_shape = args[0]->get_shape();
        int64_t seq_len         = input_shape.min_lens()[params.causal_axis];
        int64_t num_zero_slots  = std::max<int64_t>(params.state_window - seq_len, 0);
        if(num_zero_slots > 0 and input_shape.dynamic())
            MIGRAPHX_THROW("CausalConvWithState: state_window larger than the sequence length "
                           "requires a static input shape");
        std::vector<instruction_ref> slots;
        if(num_zero_slots > 0)
        {
            auto zero_lens = params.slot_lens;
            zero_lens.insert(zero_lens.begin(), num_zero_slots);
            auto zero = info.add_literal(literal{shape{input_shape.type()}, {0.0f}});
            slots.push_back(
                info.add_instruction(make_op("multibroadcast", {{"out_lens", zero_lens}}), zero));
        }
        std::vector<int64_t> backs(params.state_window - num_zero_slots);
        std::iota(backs.rbegin(), backs.rend(), 0);
        std::transform(backs.begin(), backs.end(), std::back_inserter(slots), [&](int64_t back) {
            auto slot = slice_state(info, params, seq, back);
            return info.add_instruction(make_op("unsqueeze", {{"axes", {0}}}), slot);
        });
        if(slots.size() == 1)
            return slots.front();
        return info.add_instruction(make_op("concat", {{"axis", 0}}), slots);
    }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       const onnx_parser& parser,
                                       const onnx_parser::node_info& info,
                                       const std::vector<instruction_ref>& args) const
    {
        causal_conv_parameters params;
        parse_attributes(parser, info, params);
        parse_inputs(args, params);

        auto seq    = insert_sequence(info, args, params);
        auto output = insert_output(info, args, params, seq);
        if(info.num_outputs < 2)
            return {output};
        return {output, insert_present_state(info, args, params, seq)};
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
