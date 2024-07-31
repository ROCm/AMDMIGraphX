/*
 * The MIT License (MIT)
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

#include <map>
#include <migraphx/algorithm.hpp>
#include <migraphx/common.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/lexing.hpp>
#include <migraphx/op/builder/op_builder.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

struct gemm : op_builder<gemm>
{
    float alpha  = 1.0f;
    float beta   = 1.0f;
    bool trans_a = false;
    bool trans_b = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.alpha, "alpha"),
                    f(self.beta, "beta"),
                    f(self.trans_a, "transA"),
                    f(self.trans_b, "transB"));
    }

    static std::string name() { return "gemm"; }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto a_arg = args[0];
        auto b_arg = args[1];
        if(a_arg->get_shape().ndim() != 2 or b_arg->get_shape().ndim() != 2)
        {
            MIGRAPHX_THROW("PARSE_GEMM: A and B should be rank 2, A is rank " +
                           std::to_string(a_arg->get_shape().ndim()) + ", B is rank " +
                           std::to_string(b_arg->get_shape().ndim()));
        }

        std::vector<int64_t> perm = {1, 0};
        auto dot_type             = a_arg->get_shape().type();
        if(alpha != 1.0f)
        {
            auto alpha_literal = m.add_literal(alpha);
            a_arg = migraphx::insert_common_op(m, ins, make_op("mul"), {alpha_literal, a_arg});

            if(a_arg->get_shape().type() != dot_type)
            {
                a_arg = m.insert_instruction(
                    ins, make_op("convert", {{"target_type", dot_type}}), a_arg);
            }
        }

        a_arg =
            (trans_a)
                ? m.insert_instruction(ins, make_op("transpose", {{"permutation", perm}}), a_arg)
                : a_arg;
        b_arg =
            (trans_b)
                ? m.insert_instruction(ins, make_op("transpose", {{"permutation", perm}}), args[1])
                : args[1];

        auto dot_ins = m.insert_instruction(ins, make_op("dot"), a_arg, b_arg);

        if(args.size() == 3)
        {
            if(not float_equal(beta, 0.0f))
            {
                auto c_arg = args[2];
                if(dot_ins->get_shape().dynamic())
                {
                    c_arg = m.insert_instruction(ins, make_op("multibroadcast"), args[2], dot_ins);
                }
                else
                {
                    auto out_lens   = a_arg->get_shape().lens();
                    out_lens.back() = b_arg->get_shape().lens().back();
                    auto c_lens     = c_arg->get_shape().lens();
                    if(not std::equal(
                           out_lens.begin(), out_lens.end(), c_lens.begin(), c_lens.end()))
                    {
                        c_arg = m.insert_instruction(
                            ins, make_op("multibroadcast", {{"out_lens", out_lens}}), args[2]);
                    }
                }

                if(not float_equal(beta, 1.0f))
                {
                    auto beta_literal = m.add_literal(beta);
                    c_arg =
                        migraphx::insert_common_op(m, ins, make_op("mul"), {c_arg, beta_literal});
                    if(c_arg->get_shape().type() != dot_type)
                    {
                        c_arg = m.insert_instruction(
                            ins, make_op("convert", {{"target_type", dot_type}}), c_arg);
                    }
                }

                return {m.insert_instruction(ins, make_op("add"), dot_ins, c_arg)};
            }
        }
        return {dot_ins};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
