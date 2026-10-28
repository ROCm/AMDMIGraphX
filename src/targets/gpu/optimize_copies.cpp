/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/optimize_copies.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/module.hpp>
#include <algorithm>
#include <iterator>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

namespace {

struct find_redundant_copy : match::supports_dynamic_shapes
{
    auto matcher() const
    {
        return match::any_of(
            match::name("hip::copy_from_gpu")(match::arg(0)(match::name("hip::copy_to_gpu"))),
            match::name("hip::copy_to_gpu")(match::arg(0)(match::name("hip::copy_from_gpu"))));
    }

    static bool has_internal_destination(instruction_ref copy)
    {
        if(copy->inputs().size() == 1)
            return true;
        if(copy->inputs().size() != 2)
            return false;

        // A write to a caller-owned destination is externally observable.
        const auto destination = copy->inputs().back();
        return (destination->name() == "allocate" or destination->name() == "hip::allocate") and
               destination->outputs().size() == 1;
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto copy        = r.result;
        auto source_copy = copy->inputs().front();
        auto source      = source_copy->inputs().front();
        if(has_internal_destination(source_copy) and has_internal_destination(copy) and
           copy->get_shape() == source->get_shape())
            m.replace_instruction(copy, source);
    }
};

struct find_copy_from_consumer : match::supports_dynamic_shapes
{
    static bool syncs_copies(instruction_ref ins, const std::vector<instruction_ref>& copied_inputs)
    {
        return std::any_of(ins->inputs().begin(), ins->inputs().end(), [&](instruction_ref input) {
            if(input->name() != "hip::sync_stream")
                return false;
            return std::all_of(
                copied_inputs.begin(), copied_inputs.end(), [&](instruction_ref copied_input) {
                    return std::find(input->inputs().begin(),
                                     input->inputs().end(),
                                     copied_input) != input->inputs().end();
                });
        });
    }

    auto matcher() const
    {
        return match::any(match::none_of(match::name("hip::sync_stream")),
                          match::any_of[match::inputs()](match::name("hip::copy_from_gpu")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins = r.result;
        std::vector<instruction_ref> copied_inputs;
        std::copy_if(ins->inputs().begin(),
                     ins->inputs().end(),
                     std::back_inserter(copied_inputs),
                     [](instruction_ref input) { return input->name() == "hip::copy_from_gpu"; });
        if(syncs_copies(ins, copied_inputs))
            return;

        auto sync = m.insert_instruction(ins, make_op("hip::sync_stream"), copied_inputs);
        instruction::replace_argument(ins, copied_inputs.front(), sync);
    }
};

struct find_copy_from_implicit_return : match::supports_dynamic_shapes
{
    auto matcher() const { return match::name("hip::copy_from_gpu")(match::used_once()); }

    void apply(module& m, const match::matcher_result& r) const
    {
        if(r.result->outputs().empty())
            m.add_instruction(make_op("hip::sync_stream"), r.result);
    }
};

} // namespace

void optimize_copies::apply(module& m) const
{
    match::find_matches(m, find_redundant_copy{});
    match::find_matches(m, find_copy_from_consumer{});
    match::find_matches(m, find_copy_from_implicit_return{});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
