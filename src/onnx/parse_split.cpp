/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_split : op_parser<parse_split>
{
    std::vector<op_desc> operators() const { return {{"Split"}}; }

    std::vector<instruction_ref> parse(const op_desc& opd,
                                       const onnx_parser& parser,
                                       onnx_parser::node_info info,
                                       std::vector<instruction_ref> args) const
    {
        int64_t axis = 0;
        if(contains(info.attributes, "axis"))
        {
            axis = parser.parse_value(info.attributes.at("axis")).at<int>();
        }

        auto lens          = args[0]->get_shape().lens();
        int64_t n_rank     = lens.size();
        int64_t tuned_axis = tune_axis(n_rank, axis, opd.op_name);

        std::vector<int64_t> vec_splits;
        if(contains(info.attributes, "split"))
        {
            literal s = parser.parse_value(info.attributes.at("split"));
            s.visit([&](auto v) { vec_splits.assign(v.begin(), v.end()); });

            if(std::accumulate(vec_splits.begin(), vec_splits.end(), int64_t(0)) !=
               static_cast<int64_t>(lens[tuned_axis]))
            {
                MIGRAPHX_THROW("PARSE_SPLIT: sum of split attribute unequal to dim size of axis!");
            }
        }
        // no split attribute, input is equally divided
        else
        {
            if((lens[tuned_axis] % info.num_outputs) != 0)
            {
                MIGRAPHX_THROW("PARSE_SPLIT: input cannot be equally divided into " +
                               std::to_string(info.num_outputs) + " splits!");
            }
            auto dl = lens[tuned_axis] / info.num_outputs;
            vec_splits.resize(info.num_outputs, dl);
        }

        std::vector<instruction_ref> ret_ins;
        int64_t start = 0;
        for(auto sl : vec_splits)
        {
            ret_ins.push_back(info.add_instruction(
                make_op("slice", {{"axes", {axis}}, {"starts", {start}}, {"ends", {start + sl}}}),
                args[0]));
            start += sl;
        }

        return ret_ins;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
