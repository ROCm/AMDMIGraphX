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
#include <migraphx/tf/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct parse_addn : op_parser<parse_addn>
{
    bool transpose() const { return true; }
    std::vector<op_desc> operators() const { return {{"AddN"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const tf_parser& /*parser*/,
                          const tf_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        if(args.size() == 1)
            return args[0];

        if(args.size() < 5) // using heuristic when args exceed over 5 elements
        {
            instruction_ref sum = args[0];
            for(auto i = 1; i < args.size(); i++)
            {
                sum = info.add_common_op("add", sum, args[i]);
            }
            return sum;
        }
        else
        {
            std::vector<instruction_ref> unsqueezed_args;
            std::transform(args.begin(),
                           args.end(),
                           std::back_inserter(unsqueezed_args),
                           [&info](instruction_ref arg) {
                               return info.add_instruction(make_op("unsqueeze", {{"axes", {0}}}),
                                                           arg);
                           });
            auto concatenated =
                info.add_instruction(make_op("concat", {{"axis", 0}}), unsqueezed_args);
            auto reduced =
                info.add_instruction(make_op("reduce_sum", {{"axes", {0}}}), concatenated);
            return info.add_instruction(make_op("squeeze", {{"axes", {0}}}), reduced);
        }
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
