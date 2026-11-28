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
#include <migraphx/split_sym/analyzer.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace split_sym {
namespace {

struct analyze_gathernd : analyzer<analyze_gathernd>
{
    bool matches(const operation& op) const { return op.name() == "gathernd"; }

    symbolic_op_info analyze(instruction_ref ins) const
    {
        auto inputs = to_shapes(ins->inputs());
        if(inputs.size() != 2 or inputs.back().ndim() == 0)
            return symbolic_op_info{ins};
        auto index_depth = sym::fixed_value(inputs.back().to_symbolic().dyn_dims().back().sym_expr);
        if(not index_depth.has_value())
            return symbolic_op_info{ins};
        auto batch_dims = ins->get_operator().to_value().at("batch_dims").to<int64_t>();
        if(batch_dims < 0)
            return symbolic_op_info{ins};
        std::size_t batch_rank = batch_dims;
        auto depth             = sym::to<std::size_t>(*index_depth);
        if(batch_rank + depth > inputs.front().ndim())
            return symbolic_op_info{ins};
        return analyze_axes(ins, [&](std::size_t input, std::size_t axis) {
            if(input == 1)
                return axis + 1 == inputs.back().ndim() ? axis_desc{} : parallel_axis();
            return axis >= batch_rank and axis < batch_rank + depth ? axis_desc{} : parallel_axis();
        });
    }
};

} // namespace
} // namespace split_sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
