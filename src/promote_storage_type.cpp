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
#include <migraphx/promote_storage_type.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/eliminate_convert.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/optional.hpp>
#include <algorithm>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static bool is_computation(instruction_ref ins)
{
    // convert is the storage boundary itself, and layout and identity dont
    // compute anything even though they carry the pointwise attribute
    if(contains({"convert", "layout", "identity"}, ins->name()))
        return false;
    return contains(ins->name(), "reduce") or
           ins->get_operator().attributes().get("pointwise", false);
}

static bool is_shape_transform(instruction_ref ins)
{
    static const std::unordered_set<std::string> names = {"reshape",
                                                          "squeeze",
                                                          "unsqueeze",
                                                          "flatten",
                                                          "transpose",
                                                          "broadcast",
                                                          "multibroadcast",
                                                          "contiguous"};
    return ins->inputs().size() == 1 and contains(names, ins->name());
}

void promote_storage_type::apply(module_pass_manager& mpm) const
{
    auto& m = mpm.get_module();
    // The converts to the storage type inserted after each promoted
    // instruction
    std::unordered_set<instruction_ref> storage_converts;
    // When the input is a promoted value converted back to the storage type,
    // possibly viewed through shape ops, rebuild the view on the float value
    // instead so there is no convert back and forth between adjacent promoted
    // instructions
    auto promoted_input = [&](instruction_ref ins,
                              instruction_ref input) -> optional<instruction_ref> {
        std::vector<instruction_ref> views;
        auto x = input;
        while(is_shape_transform(x))
        {
            views.push_back(x);
            x = x->inputs().front();
        }
        if(not contains(storage_converts, x))
            return nullopt;
        auto result = x->inputs().front();
        std::for_each(views.rbegin(), views.rend(), [&](instruction_ref view) {
            result = m.insert_instruction(ins, view->get_operator(), result);
        });
        return result;
    };
    for(auto ins : iterator_for(m))
    {
        if(not contains(types, ins->get_shape().type()))
            continue;
        if(not is_computation(ins))
            continue;
        auto convert_back = m.insert_instruction(
            std::next(ins), make_op("convert", {{"target_type", ins->get_shape().type()}}), ins);
        m.replace_instruction(ins, convert_back);
        std::vector<instruction_ref> inputs;
        std::transform(
            ins->inputs().begin(),
            ins->inputs().end(),
            std::back_inserter(inputs),
            [&](instruction_ref input) {
                if(not contains(types, input->get_shape().type()))
                    return input;
                if(auto promoted = promoted_input(ins, input))
                    return *promoted;
                return m.insert_instruction(
                    ins, make_op("convert", {{"target_type", shape::float_type}}), input);
            });
        m.replace_instruction(ins, ins->get_operator(), inputs);
        storage_converts.insert(convert_back);
    }
    // Adjacent promoted instructions are connected by a convert to the
    // storage type followed by a convert back to float, which cancel
    mpm.run_pass(eliminate_convert{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
