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

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto l0      = args[0];
        auto l1      = args[1];
        auto l0_lens = l0->get_shape().lens();
        auto l1_lens = l1->get_shape().lens();

        // args[0] is a vector, prepend 1 to the shape
        bool is_a_prepended = false;
        if(l0_lens.size() == 1)
        {
            is_a_prepended = true;
            l0_lens.insert(l0_lens.begin(), 1);
            l0 = info.add_instruction(make_op("unsqueeze", {{"axes", {0}}}), args[0]);
        }

        bool is_b_appended = false;
        if(l1_lens.size() == 1)
        {
            is_b_appended = true;
            l1_lens.push_back(1);
            l1 = info.add_instruction(make_op("unsqueeze", {{"axes", {1}}}), args[1]);
        }

        instruction_ref bl0 = l0;
        instruction_ref bl1 = l1;
        if(not std::equal(
               l0_lens.rbegin() + 2, l0_lens.rend(), l1_lens.rbegin() + 2, l1_lens.rend()))
        {
            auto l0_it = l0_lens.begin() + l0_lens.size() - 2;
            std::vector<std::size_t> l0_broadcasted_lens(l0_lens.begin(), l0_it);
            auto l1_it = l1_lens.begin() + l1_lens.size() - 2;
            std::vector<std::size_t> l1_broadcasted_lens(l1_lens.begin(), l1_it);
            auto output_lens = compute_broadcasted_lens(l0_broadcasted_lens, l1_broadcasted_lens);
            l0_broadcasted_lens = output_lens;
            l0_broadcasted_lens.insert(l0_broadcasted_lens.end(), l0_it, l0_lens.end());
            l1_broadcasted_lens = output_lens;
            l1_broadcasted_lens.insert(l1_broadcasted_lens.end(), l1_it, l1_lens.end());
            if(l0_lens != l0_broadcasted_lens)
            {
                bl0 = info.add_instruction(
                    make_op("multibroadcast", {{"out_lens", l0_broadcasted_lens}}), l0);
            }
            if(l1_lens != l1_broadcasted_lens)
            {
                bl1 = info.add_instruction(
                    make_op("multibroadcast", {{"out_lens", l1_broadcasted_lens}}), l1);
            }
        }
        instruction_ref dot_res = info.add_instruction(make_op(opd.op_name), bl0, bl1);
        int64_t num_axis        = static_cast<int64_t>(dot_res->get_shape().lens().size());
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
