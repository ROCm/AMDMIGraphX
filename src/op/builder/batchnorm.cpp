/* The MIT License (MIT)
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

#include <migraphx/common.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/op/builder/insert.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

struct batchnorm : op_builder<batchnorm>
{
    float epsilon = 1e-5f;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.epsilon, "epsilon"));
    }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto x_lens = args[0]->get_shape().max_lens();
        auto x_type = args[0]->get_shape().type();

        if(std::any_of(args.cbegin() + 1, args.cend(), [](auto a) {
               return a->get_shape().lens().size() != 1;
           }))
        {
            MIGRAPHX_THROW("batchnorm op_builder: argument scale, bias, mean, or var rank != 1");
        }

        auto x_rank = x_lens.size();
        if(x_rank == 1 or x_rank == 2)
        {
            auto eps =
                m.add_literal(migraphx::literal{migraphx::shape{x_type}, {epsilon}}, debug_symbols);
            auto x_sub_mean = insert_common_op(m, ins, "sub", debug_symbols, args[0], args[3]);
            auto var_eps    = insert_common_op(m, ins, "add", debug_symbols, args[4], eps);
            auto rsqrt      = m.insert_instruction(ins, make_op("rsqrt"), debug_symbols, var_eps);
            auto mul0       = insert_common_op(m, ins, "mul", debug_symbols, args[1], rsqrt);
            auto r0         = insert_common_op(m, ins, "mul", debug_symbols, x_sub_mean, mul0);
            return {insert_common_op(m, ins, "add", debug_symbols, r0, args[2])};
        }
        else if(x_rank > 2)
        {
            // unsqueeze tensors of shape (C) to broadcast correctly
            std::vector<int64_t> unsqueeze_axes(x_lens.size() - 2);
            std::iota(unsqueeze_axes.begin(), unsqueeze_axes.end(), 1);
            auto eps =
                m.add_literal(migraphx::literal{migraphx::shape{x_type}, {epsilon}}, debug_symbols);
            auto scale_unsqueeze =
                m.insert_instruction(ins,
                                     migraphx::make_op("unsqueeze", {{"axes", unsqueeze_axes}}),
                                     debug_symbols,
                                     args[1]);
            auto bias_unsqueeze =
                m.insert_instruction(ins,
                                     migraphx::make_op("unsqueeze", {{"axes", unsqueeze_axes}}),
                                     debug_symbols,
                                     args[2]);
            auto mean_unsqueeze =
                m.insert_instruction(ins,
                                     migraphx::make_op("unsqueeze", {{"axes", unsqueeze_axes}}),
                                     debug_symbols,
                                     args[3]);
            auto var_unsqueeze =
                m.insert_instruction(ins,
                                     migraphx::make_op("unsqueeze", {{"axes", unsqueeze_axes}}),
                                     debug_symbols,
                                     args[4]);
            auto x_sub_mean =
                insert_common_op(m, ins, "sub", debug_symbols, args[0], mean_unsqueeze);
            auto var_eps = insert_common_op(m, ins, "add", debug_symbols, var_unsqueeze, eps);
            auto rsqrt   = m.insert_instruction(ins, make_op("rsqrt"), debug_symbols, var_eps);
            auto mul0    = insert_common_op(m, ins, "mul", debug_symbols, scale_unsqueeze, rsqrt);
            auto r0      = insert_common_op(m, ins, "mul", debug_symbols, x_sub_mean, mul0);
            return {insert_common_op(m, ins, "add", debug_symbols, r0, bias_unsqueeze)};
        }
        else
        {
            // rank == 0
            MIGRAPHX_THROW("batchnorm op_builder: rank " + std::to_string(x_lens.size()) +
                           " input tensor, unhandled data format");
        }
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
