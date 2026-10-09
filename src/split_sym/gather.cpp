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

bool has_nonnegative_values(instruction_ref ins)
{
    const auto& inputs = ins->inputs();
    if(ins->name() == "@literal")
    {
        bool result = true;
        ins->get_literal().visit(
            [&](auto values) { result = all_of(values, [](auto value) { return value >= 0; }); });
        return result;
    }
    if(ins->name() == "fill")
        return not inputs.empty() and has_nonnegative_values(inputs.front());
    if(contains({"convert",
                 "contiguous",
                 "reshape",
                 "squeeze",
                 "unsqueeze",
                 "transpose",
                 "slice",
                 "dyn_slice",
                 "broadcast",
                 "multibroadcast"},
                ins->name()))
        return not inputs.empty() and has_nonnegative_values(inputs.front());
    if(ins->name() == "concat")
        return not inputs.empty() and all_of(inputs, has_nonnegative_values);
    if(ins->name() == "gather")
        return not inputs.empty() and has_nonnegative_values(inputs.front());
    if(contains({"add", "mul"}, ins->name()))
        return not inputs.empty() and all_of(inputs, has_nonnegative_values);
    if(ins->name() == "eval_expr_from_shape")
        return true;
    if(ins->name() != "get_tuple_elem" or inputs.size() != 1)
        return false;

    auto index  = ins->get_operator().to_value().at("index").to<std::size_t>();
    auto source = inputs.front();
    if(source->name() == "dyn_concat" and source->inputs().size() % 2 == 0)
    {
        if(index == 1)
            return true;
        auto data_inputs = source->inputs();
        data_inputs.resize(data_inputs.size() / 2);
        return index == 0 and all_of(data_inputs, has_nonnegative_values);
    }
    return (source->name() == "topk" and index == 1) or
           (contains({"nonzero", "nonmaxsuppression"}, source->name()) and index == 0);
}

struct analyze_gather : analyzer<analyze_gather>
{
    bool matches(const operation& op) const { return op.name() == "gather"; }

    symbolic_op_info analyze(instruction_ref ins) const
    {
        auto inputs = to_shapes(ins->inputs());
        if(inputs.size() != 2)
            return symbolic_op_info{ins};
        auto axis = normalize_axis(ins->get_operator().to_value().at("axis").to<int64_t>(),
                                   inputs.front().ndim());
        if(not axis.has_value())
            return symbolic_op_info{ins};
        bool nonnegative_indices = has_nonnegative_values(ins->inputs().at(1));
        // Padded index entries are zero and their gathered outputs are sliced away. Negative
        // indices are relative to the unpadded extent, so they cannot use this rewrite.
        return analyze_axes(
            ins, [axis = *axis, nonnegative_indices](std::size_t input, std::size_t current_axis) {
                if(input == 1 or current_axis != axis)
                    return parallel_axis();
                return nonnegative_indices ? contracted_axis(fill_kind::dont_care) : axis_desc{};
            });
    }
};

} // namespace
} // namespace split_sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
