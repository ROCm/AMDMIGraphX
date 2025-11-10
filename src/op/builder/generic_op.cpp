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

#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

struct generic_op : op_builder<generic_op>
{
    operation op;

    void from_value(const value& v) { op = migraphx::from_value<operation>(v); }

    static std::vector<std::string> names()
    {
        return {"abs",      "acos",  "acosh",      "asin",  "asinh",   "atan",
                "atanh",    "ceil",  "concat",     "cos",   "cosh",    "elu",
                "erf",      "exp",   "flatten",    "floor", "gather",  "gathernd",
                "identity", "isnan", "leaky_relu", "log",   "lrn",     "neg",
                "recip",    "relu",  "nearbyint",  "rsqrt", "sigmoid", "sign",
                "sin",      "sinh",  "sqrt",       "tan",   "tanh",    "not"};
    }

    std::vector<instruction_ref> insert(const std::string& op_name,
                                        module& m,
                                        instruction_ref /*ins*/,
                                        const std::vector<instruction_ref>& args) const
    {
        std::vector<instruction_ref> args_copy = args;
        if(needs_contiguous(op_name))
        {
            std::transform(args_copy.begin(), args_copy.end(), args_copy.begin(), [&](auto arg) {
                return make_contiguous(m, arg);
            });
        }

        return {m.add_instruction(op, args_copy)};
    }

    private:
    bool needs_contiguous(const std::string& op_name) const
    {
        return contains({"flatten", "gather", "scatter"}, op_name);
    }

    instruction_ref make_contiguous(module& m, instruction_ref ins) const
    {
        auto attr       = ins->get_operator().to_value();
        std::string key = "require_std_shape";
        if((attr.get(key, false)) or (not ins->get_shape().standard()))
        {
            return m.add_instruction(make_op("contiguous"), ins);
        }

        return ins;
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
