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

#include <migraphx/common.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/op/builder/insert.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

struct arg_op : op_builder<arg_op>
{
    int64_t axis           = 0;
    int keep_dims          = 1;
    bool select_last_index = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"),
                    f(self.keep_dims, "keepdims"),
                    f(self.select_last_index, "select_last_index"));
    }

    static std::vector<std::string> names() { return {"argmax", "argmin"}; }

    std::vector<instruction_ref> insert(const std::string& op_name,
                                        module& m,
                                        instruction_ref ins,
                                        const std::vector<instruction_ref>& args) const
    {
        if(keep_dims == 0)
        {
            auto arg_ins = m.insert_instruction(
                ins,
                make_op(op_name, {{"axis", axis}, {"select_last_index", select_last_index}}),
                args);
            return {m.insert_instruction(ins, make_op("squeeze", {{"axes", {axis}}}), arg_ins)};
        }
        else
        {
            return {m.insert_instruction(
                ins,
                make_op(op_name, {{"axis", axis}, {"select_last_index", select_last_index}}),
                args)};
        }
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
