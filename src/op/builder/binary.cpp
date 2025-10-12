/*
 * The MIT License (MIT)
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

#include <migraphx/instruction.hpp>
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/common.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

struct binary : op_builder<binary>
{
    uint64_t broadcasted = 0;
    uint64_t axis = 0;
    bool is_broadcasted = false;

    static std::vector<std::string> names() { return {"add", "div", "logical_and", "logical_or", "logical_xor", "bitwise_and", "mul", "prelu", "sub"}; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.broadcasted, "broadcasted"),
                    f(self.axis, "axis"),
                    f(self.is_broadcasted, "is_broadcasted"));
    }

    std::vector<instruction_ref>
    insert(const std::string& op_name, module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        if (is_broadcasted)
        {
            if (broadcasted != 0)
            {
                if(std::any_of(args.cbegin(), args.cend(), [](auto a) { return a->get_shape().dynamic(); }))
                {
                    MIGRAPHX_THROW("Binary op broadcast attribute not supported for dynamic input shapes");
                }
                auto l = m.add_instruction(migraphx::make_op("broadcast",{{"axis", axis}, {"out_lens", args[0]->get_shape().lens()}}),args[1]);
                return {m.add_instruction(migraphx::make_op(op_name), args[0], l)};
            }
            return {m.add_instruction(migraphx::make_op(op_name), args)};
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
