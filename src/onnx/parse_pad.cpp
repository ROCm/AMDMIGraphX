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
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/pad.hpp>
#include <migraphx/sym.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_pad : op_parser<parse_pad>
{
    std::vector<op_desc> operators() const { return {{"Pad"}}; }

    std::string parse_mode(const onnx_parser::node_info& info,
                           const std::vector<instruction_ref>& args) const
    {
        if(contains(info.attributes, "mode"))
        {
            auto mode = info.attributes.at("mode").s();
            if(mode == "reflect" or mode == "edge")
            {
                if(args.front()->get_shape().dynamic())
                {
                    MIGRAPHX_THROW("PARSE_PAD: " + mode +
                                   " padding with dynamic shape not supported");
                }
            }
            else if(mode != "constant" and mode != "edge")
            {
                MIGRAPHX_THROW("PARSE_PAD: MIGraphX currently only supports constant, reflect, and "
                               "edge padding");
            }
            return mode;
        }
        else
        {
            // default mode
            return "constant";
        }
    }

    std::vector<int64_t> parse_pads(const onnx_parser::node_info& info,
                                    const std::vector<instruction_ref>& args) const
    {
        std::vector<int64_t> pads{};
        if(args.size() >= 2)
        {
            auto pad_arg = args.at(1)->eval();
            check_arg_empty(pad_arg, "PARSE_PAD: `pads` input must be constant");
            pad_arg.visit([&](auto v) { pads.assign(v.begin(), v.end()); });
        }
        else if(contains(info.attributes, "pads"))
        {
            auto&& pad_vals = info.attributes.at("pads").ints();
            pads            = std::vector<int64_t>(pad_vals.begin(), pad_vals.end());
        }
        else
        {
            MIGRAPHX_THROW("PARSE_PAD: `pads` must be available");
        }
        return pads;
    }

    float parse_constant_value(const onnx_parser& parser,
                               const onnx_parser::node_info& info,
                               const std::vector<instruction_ref>& args) const
    {
        float value = 0.0f;
        if(args.size() >= 3 and args.at(2)->get_shape().elements() >= 1)
        {
            auto val_ins = args.at(2);
            if(not val_ins->can_eval())
            {
                MIGRAPHX_THROW("PARSE_PAD: input `value` must be constant");
            }
            auto val_arg = val_ins->eval();
            if(val_arg.get_shape().elements() != 1)
            {
                MIGRAPHX_THROW("PARSE_PAD: `value` should contain only one element");
            }
            value = val_arg.at<float>();
        }
        else if(contains(info.attributes, "value"))
        {
            value = parser.parse_value(info.attributes.at("value")).at<float>();
        }
        return value;
    }

    std::vector<int64_t> parse_axes(const std::vector<instruction_ref>& args,
                                    bool is_constant_mode) const
    {
        std::vector<int64_t> axes{};
        // axes is 3rd or 4th, depending on constant mode
        auto pos = is_constant_mode ? 4 : 3;
        if(args.size() >= pos)
        {
            auto axes_arg = args.at(pos - 1)->eval();
            check_arg_empty(axes_arg, "PARSE_PAD: variable `axes` input not supported");
            axes_arg.visit([&](auto v) { axes.assign(v.begin(), v.end()); });
        }
        return axes;
    }

    std::vector<int64_t> calculate_pads_with_axes(const std::vector<int64_t>& pads,
                                                  const std::vector<int64_t>& axes,
                                                  size_t input_rank) const
    {
        size_t num_axes = axes.size();
        if(num_axes * 2 != pads.size())
        {
            MIGRAPHX_THROW("PARSE_PAD: number of elements of pads should be equal to 2 * "
                           "number of elements of axes");
        }

        std::vector<int64_t> new_pads(input_rank * 2);
        for(size_t idx{0}; idx < num_axes; ++idx)
        {
            // axis can be negative
            int64_t axis = axes[idx] < 0 ? input_rank + axes[idx] : axes[idx];
            // pad format is x1_begin, x2_begin, ... , x3_end, x4_end
            new_pads[axis]              = pads[idx];
            new_pads[axis + input_rank] = pads[idx + num_axes];
        }
        return new_pads;
    }

    // `pads` that are not constant but known as expressions over the symbols of the program: pad
    // by concatenating the fill value, broadcast to the run-time pad size, along each padded axis
    instruction_ref parse_symbolic_pads(const onnx_parser& parser,
                                        const onnx_parser::node_info& info,
                                        const std::vector<instruction_ref>& args,
                                        const std::vector<sym::expr>& pads) const
    {
        auto x            = args.front();
        size_t input_rank = x->get_shape().ndim();
        if(x->get_shape().dynamic() and not x->get_shape().symbolic())
            MIGRAPHX_THROW("PARSE_PAD: non-constant `pads` need a static or symbolic input shape");
        if(parse_mode(info, args) != "constant")
            MIGRAPHX_THROW("PARSE_PAD: non-constant `pads` are only supported in constant mode");
        if(not parse_axes(args, true).empty())
            MIGRAPHX_THROW("PARSE_PAD: non-constant `pads` are not supported with `axes`");
        if(pads.size() != input_rank * 2)
        {
            MIGRAPHX_THROW("PARSE_PAD: number of elements of pads should be equal to 2 * "
                           "input rank");
        }

        const auto zero = sym::lit(int64_t{0});
        auto fill =
            info.add_literal(literal{shape{x->get_shape().type()},
                                     std::vector<float>{parse_constant_value(parser, info, args)}});
        auto slice_of = [&](instruction_ref ins, std::size_t start, std::size_t end) {
            return info.add_instruction(
                make_op("slice", {{"axes", {0}}, {"starts", {start}}, {"ends", {end}}}), ins);
        };
        for(std::size_t axis = 0; axis < input_rank; axis++)
        {
            for(auto at_end : {false, true})
            {
                auto index      = axis + (at_end ? input_rank : 0);
                const auto& pad = pads[index];
                if(pad == zero)
                    continue;
                if(sym::strict_less(pad, zero).value_or(false))
                    MIGRAPHX_THROW("PARSE_PAD: non-constant `pads` cannot be negative");
                // The run-time dims of the fill: those of x, with the pad size on this axis
                auto x_dims =
                    info.add_instruction(make_op("dimensions_of", {{"end", input_rank}}), x);
                std::vector<instruction_ref> parts;
                if(axis > 0)
                    parts.push_back(slice_of(x_dims, 0, axis));
                parts.push_back(slice_of(args.at(1), index, index + 1));
                if(axis + 1 < input_rank)
                    parts.push_back(slice_of(x_dims, axis + 1, input_rank));
                auto fill_dims =
                    parts.size() == 1
                        ? parts.front()
                        : info.add_instruction(make_op("concat", {{"axis", 0}}), parts);
                auto dims     = x->get_shape().to_symbolic().dyn_dims();
                dims.at(axis) = shape::dynamic_dimension{pad};
                auto filled   = info.add_instruction(
                    make_op("broadcast_with_dims", {{"out_dyn_dims", to_value(dims)}}),
                    fill,
                    fill_dims);
                std::vector<instruction_ref> inputs{filled, x};
                if(at_end)
                    std::swap(inputs.front(), inputs.back());
                x = info.add_instruction(make_op("concat", {{"axis", axis}}), inputs);
            }
        }
        return x;
    }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        if(args.size() >= 2 and args.at(1)->eval().empty())
        {
            const auto pads_value = args.at(1)->sym_eval();
            if(not pads_value.empty())
            {
                const auto expressions = pads_value.get();
                return parse_symbolic_pads(
                    parser, info, args, {expressions.begin(), expressions.end()});
            }
        }
        std::vector<int64_t> pads = parse_pads(info, args);

        // check if padding is actually being done (at least one value is nonzero)
        if(std::all_of(pads.begin(), pads.end(), [](const int& i) { return i == 0; }))
        {
            return info.add_instruction(make_op("identity"), args.front());
        }

        std::string mode      = parse_mode(info, args);
        bool is_constant_mode = mode == "constant";
        float value           = is_constant_mode ? parse_constant_value(parser, info, args) : 0.0f;
        std::vector<int64_t> axes = parse_axes(args, is_constant_mode);
        size_t input_rank         = args.front()->get_shape().ndim();

        if(not axes.empty())
        {
            pads = calculate_pads_with_axes(pads, axes, input_rank);
        }

        if(pads.size() != input_rank * 2)
        {
            MIGRAPHX_THROW("PARSE_PAD: number of elements of pads should be equal to 2 * "
                           "input rank");
        }

        auto mode_val = op::pad::constant_pad;
        if(mode == "reflect")
        {
            mode_val = op::pad::reflect_pad;
        }
        else if(mode == "edge")
        {
            mode_val = op::pad::edge_pad;
        }

        return info.add_instruction(
            migraphx::make_op("pad", {{"pads", pads}, {"value", value}, {"mode", mode_val}}),
            args.front());
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
