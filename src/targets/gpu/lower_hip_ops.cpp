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
#include <migraphx/gpu/lower_hip_ops.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/value.hpp>
#include <algorithm>
#include <cassert>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

namespace {
operation precompiled(const operation& op)
{
    // additional_args == 0 since hip::fill/hip::copy already include their output buffer as an
    // input
    return make_op("gpu::precompile_op", {{"op", to_value(op)}, {"additional_args", 0}});
}

struct find_hip_memory_op
{
    auto matcher() const { return match::name("hip::fill", "hip::copy"); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins = r.result;
        if(ins->get_shape().dynamic())
            return;
        assert(not ins->inputs().empty());
        auto pre = precompiled(ins->get_operator());

        const auto& subs = ins->get_shape().sub_shapes();
        if(subs.empty())
        {
            m.replace_instruction(ins, pre, ins->inputs());
            return;
        }

        assert(std::all_of(ins->inputs().begin(), ins->inputs().end(), [&](auto in) {
            return in->get_shape().sub_shapes().size() == subs.size();
        }));

        // A code object handles one tensor, so a tuple buffer is filled/copied per sub-object
        std::vector<instruction_ref> elems = {ins->inputs().back()};
        for(auto i : range(subs.size()))
        {
            std::vector<instruction_ref> sub_inputs;
            std::transform(ins->inputs().begin(),
                           ins->inputs().end(),
                           std::back_inserter(sub_inputs),
                           [&](auto in) {
                               return m.insert_instruction(
                                   ins, make_op("get_tuple_elem", {{"index", i}}), in);
                           });
            elems.push_back(m.insert_instruction(ins, pre, sub_inputs));
        }
        m.replace_instruction(ins, make_op("identity"), elems);
    }
};
} // namespace

void lower_hip_ops::apply(module& m) const { match::find_matches(m, find_hip_memory_op{}); }

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
