/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/tune_axis.hpp>
#include <migraphx/common.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_quantizelinear : op_parser<parse_quantizelinear>
{
    std::vector<op_desc> operators() const { return {{"QuantizeLinear"}}; }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref>& args) const
    {
        if(args.size() < 2 or args.size() > 3)
        {
            MIGRAPHX_THROW("QuantizeLinear: must have either 2 or 3 inputs, " +
                           std::to_string(args.size()) + " inputs provided");
        }

        if(args[0]->get_shape().type() != args[1]->get_shape().type())
        {
            MIGRAPHX_THROW("QuantizeLinear: x and y_scale must be of same type");
        }

        if(args.size() == 3 and args[1]->get_shape().lens() != args[2]->get_shape().lens())
        {
            MIGRAPHX_THROW(
                "QuantizeLinear: y_scale and y_zero_point shapes must be equal. Provided y_scale "
                "shape: " +
                to_string_range(args[1]->get_shape().lens()) +
                ", provided y_zero_point shape: " + to_string_range(args[2]->get_shape().lens()));
        }

        int axis = 1;
        if(contains(info.attributes, "axis"))
            axis = info.attributes.at("axis").i();

        // TODO handle block size of zero
        int block_size = 0;
        if(contains(info.attributes, "block_size"))
            block_size = info.attributes.at("block_size").i();

        const auto x          = args.at(0);
        const auto x_lens     = x->get_shape().lens();
        const auto x_rank     = x_lens.size();
        const auto tuned_axis = tune_axis(x_rank, axis, opd.op_name);

        instruction_ref y_scale = args.at(1);
        const auto y_scale_lens = y_scale->get_shape().lens();
        const auto y_scale_rank = y_scale_lens.size();

        if(y_scale->get_shape().scalar())
        {
            std::transform(args.begin() + 1, args.end(), args.begin() + 1, [&](auto ins) {
                return info.add_instruction(make_op("multibroadcast", {{"out_lens", x_lens}}), ins);
            });
        }
        else if(y_scale_rank == 1)
        {
            if(x_lens[tuned_axis] != y_scale_lens[0])
            {
                MIGRAPHX_THROW("TODO");
            }

            std::transform(args.begin() + 1, args.end(), args.begin() + 1, [&](auto ins) {
                return info.add_instruction(
                    make_op("broadcast", {{"axis", tuned_axis}, {"out_lens", x_lens}}), ins);
            });
        }
        else
        {
            if(block_size == 0)
            {
                MIGRAPHX_THROW("TODO");
            }

            if(x_rank != y_scale_rank)
            {
                MIGRAPHX_THROW("TODO");
            }

            for(auto i = 0u; i < x_lens.size(); ++i)
            {
                if(x_lens[i] != y_scale_lens[i] and i != tuned_axis)
                {
                    MIGRAPHX_THROW("TODO");
                }
            }

            // Given x shape (D0, ..., Di, ..., Dn), y_scale shape (S0, ... Si, ...Sn) and
            // axis=i, the accepted range is [ceil(Di/Si), ceil(Di/(Si-1))-1]
            float di           = x_lens[tuned_axis];
            float si           = y_scale_lens[tuned_axis];
            int block_size_min = std::ceil(di / si);
            int block_size_max = std::ceil(di / (si - 1)) - 1;
            if(block_size < block_size_min or block_size > block_size_max)
                MIGRAPHX_THROW("TODO");

            std::transform(args.begin() + 1, args.end(), args.begin() + 1, [&](auto ins) {
                if(block_size == 1)
                    return ins;

                ins = info.add_instruction(make_op("unsqueeze", {{"axes", {tuned_axis + 1}}}), ins);

                auto bc_lens            = ins->get_shape().lens();
                bc_lens[tuned_axis + 1] = block_size;
                ins = info.add_instruction(make_op("multibroadcast", {{"out_lens", bc_lens}}), ins);

                auto reshape_lens        = x_lens;
                reshape_lens[tuned_axis] = ins->get_shape().lens()[tuned_axis] * block_size;
                ins = info.add_instruction(make_op("reshape", {{"dims", reshape_lens}}), ins);

                // Detect runt block
                if(x_lens[tuned_axis] < reshape_lens[tuned_axis])
                {
                    ins = info.add_instruction(make_op("slice",
                                                       {{"axes", {tuned_axis}},
                                                        {"starts", {0}},
                                                        {"ends", {x_lens[tuned_axis]}}}),
                                               ins);
                }

                return ins;
            });
        }

        return info.add_instruction(make_op("quantizelinear"), args);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
