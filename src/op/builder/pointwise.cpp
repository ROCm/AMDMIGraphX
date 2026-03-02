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

#include <migraphx/instruction.hpp>
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/common.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

struct pointwise : op_builder<pointwise>
{
    std::optional<uint64_t> broadcasted_axis = std::nullopt;

    static std::vector<std::string> names()
    {
        return {"add",
                "div",
                "logical_and",
                "logical_or",
                "logical_xor",
                "bitwise_and",
                "mul",
                "pow",
                "prelu",
                "sqdiff",
                "sub"};
    }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.broadcasted_axis, "broadcasted_axis"));
    }

    std::vector<instruction_ref> insert(const std::string& op_name,
                                        module& m,
                                        instruction_ref ins,
                                        const std::vector<instruction_ref>& args) const
    {
        if(broadcasted_axis.has_value())
        {
            const size_t shorter_instr_idx =
                args[0]->get_shape().ndim() < args[1]->get_shape().ndim() ? 0 : 1;
            const size_t longer_instr_idx = shorter_instr_idx == 0 ? 1 : 0;

            auto l = m.add_instruction(
                migraphx::make_op("broadcast",
                                  {{"axis", broadcasted_axis.value()},
                                   {"out_lens", args[longer_instr_idx]->get_shape().lens()}}),
                args[shorter_instr_idx]);
            return {m.add_instruction(migraphx::make_op(op_name), args[longer_instr_idx], l)};
        }
        else
        {
            return {insert_common_op(m, ins, migraphx::make_op(op_name), args)};
        }
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
