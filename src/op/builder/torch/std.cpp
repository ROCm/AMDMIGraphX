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

#include <cstddef>
#include <cstdint>
#include <vector>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/tune_axis.hpp>
#include <migraphx/op/builder/insert.hpp>
#include <migraphx/op/builder/op_builder.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

// std has no native op: sqrt of the corrected variance reduced over axes.
struct torch_std : op_builder<torch_std>
{
    std::vector<int64_t> axes = {};
    bool keepdim              = false;
    float correction          = 1.0f;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(
            f(self.axes, "axes"), f(self.keepdim, "keepdim"), f(self.correction, "correction"));
    }

    static std::vector<std::string> names() { return {"tm::std"}; }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto x    = args[0];
        auto lens = x->get_shape().lens();
        auto type = x->get_shape().type();

        std::size_t n = 1;
        for(auto a : axes)
            n *= lens[tune_axis(lens.size(), a, "std")];

        auto mean  = m.insert_instruction(ins, make_op("reduce_mean", {{"axes", axes}}), x);
        auto sub   = insert_common_op(m, ins, "sub", x, mean);
        auto sq    = insert_common_op(m, ins, "mul", sub, sub);
        auto sum   = m.insert_instruction(ins, make_op("reduce_sum", {{"axes", axes}}), sq);
        auto denom = m.add_literal({type, {static_cast<float>(n) - correction}});
        auto var   = insert_common_op(m, ins, "div", sum, denom);
        auto out   = m.insert_instruction(ins, make_op("sqrt"), var);
        if(not keepdim)
            out = m.insert_instruction(ins, make_op("squeeze", {{"axes", axes}}), out);
        return {out};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
