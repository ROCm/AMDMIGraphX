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
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_gemm : op_parser<parse_gemm>
{
    std::vector<op_desc> operators() const { return {{"Gemm"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        float alpha = 1.0f;
        float beta  = 1.0f;
        bool transa = false;
        bool transb = false;
        if(contains(info.attributes, "alpha"))
        {
            alpha = parser.parse_value(info.attributes.at("alpha")).at<float>();
        }
        if(contains(info.attributes, "beta"))
        {
            beta = parser.parse_value(info.attributes.at("beta")).at<float>();
        }
        if(contains(info.attributes, "transA"))
        {
            transa = parser.parse_value(info.attributes.at("transA")).at<bool>();
        }
        if(contains(info.attributes, "transB"))
        {
            transb = parser.parse_value(info.attributes.at("transB")).at<bool>();
        }

        std::vector<int64_t> perm(args[0]->get_shape().lens().size());
        std::iota(perm.begin(), perm.end(), int64_t{0});
        // swap the last two elements
        std::swap(*perm.rbegin(), *(perm.rbegin() + 1));

        auto l1       = args[0];
        auto dot_type = l1->get_shape().type();

        if(alpha != 1.0f)
        {
            auto alpha_literal = info.add_literal(alpha);
            l1                 = info.add_broadcastable_binary_op("mul", alpha_literal, l1);
            if(l1->get_shape().type() != dot_type)
            {
                l1 = info.add_instruction(make_op("convert", {{"target_type", dot_type}}), l1);
            }
        }

        l1 =
            (transa) ? info.add_instruction(make_op("transpose", {{"permutation", perm}}), l1) : l1;
        auto l2 = (transb)
                      ? info.add_instruction(make_op("transpose", {{"permutation", perm}}), args[1])
                      : args[1];

        auto ret = info.add_instruction(make_op("dot"), l1, l2);

        if(args.size() == 3)
        {
            if(not float_equal(beta, 0.0f) && args[2]->get_shape().elements() > 0)
            {
                auto out_lens   = l1->get_shape().lens();
                out_lens.back() = l2->get_shape().lens().back();
                auto l3         = args[2];
                auto l3_lens    = l3->get_shape().lens();
                if(not std::equal(out_lens.begin(), out_lens.end(), l3_lens.begin(), l3_lens.end()))
                {
                    l3 = info.add_instruction(make_op("multibroadcast", {{"out_lens", out_lens}}),
                                              args[2]);
                }
                auto beta_literal = info.add_literal(beta);
                auto beta_l3      = info.add_broadcastable_binary_op("mul", l3, beta_literal);
                if(beta_l3->get_shape().type() != dot_type)
                {
                    beta_l3 = info.add_instruction(make_op("convert", {{"target_type", dot_type}}),
                                                   beta_l3);
                }

                return info.add_instruction(make_op("add"), ret, beta_l3);
            }
        }

        return ret;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
