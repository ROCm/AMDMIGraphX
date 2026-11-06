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
#include <migraphx/gpu/lower_reshape.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/reshape_dims.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/value.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

namespace {
instruction_ref
insert_precompile_op(module& m, instruction_ref pos, instruction_ref input, const operation& op)
{
    auto output_shape = op.compute_shape({input->get_shape()});
    auto alloc =
        m.insert_instruction(pos, make_op("allocate", {{"shape", to_value(output_shape)}}));
    return m.insert_instruction(
        pos, make_op("gpu::precompile_op", {{"op", to_value(op)}}), input, alloc);
}

instruction_ref insert_contiguous(module& m, instruction_ref pos, instruction_ref input)
{
    const auto& s      = input->get_shape();
    shape output_shape = s.dynamic() ? shape{s.type(), s.dyn_dims()} : shape{s.type(), s.lens()};
    auto alloc =
        m.insert_instruction(pos, make_op("allocate", {{"shape", to_value(output_shape)}}));
    return m.insert_instruction(pos, make_op("gpu::contiguous"), input, alloc);
}

struct find_reshape : match::supports_dynamic_shapes
{
    // Skip reshape(data, output_buffer). Every GPU copy op derives its kernel from one
    // index space shared by source and destination, so none of them can change rank.
    auto matcher() const { return match::name("reshape")(match::nargs(1)); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins        = r.result;
        auto dims       = ins->get_operator().to_value().at("dims");
        auto reshape_op = make_op("reshape_lazy", {{"dims", {dims}}});
        auto input      = ins->inputs().front();
        const auto& s   = input->get_shape();

        if(not s.dynamic() or s.symbolic())
        {
            auto expected    = ins->get_shape().to_symbolic();
            auto output_dims = ins->get_shape().sym_dims();
            auto reshaped    = reshape_dims(s.to_symbolic(), output_dims, {.lazy = true});
            if(reshaped and sym::same_symbol(reshaped->sym_elements(), expected.sym_elements()))
            {
                m.replace_instruction(ins, reshape_op, {input});
                return;
            }

            auto relayout =
                reshape_dims(ins->get_shape().to_symbolic(), s.sym_dims(), {.lazy = true});
            if(relayout)
            {
                auto perm = find_permutation(*relayout);
                // An identity permutation means the input only needs standardizing, which is
                // what gpu::contiguous already does. Prefer it: lower_device_ops turns it into
                // a prebuilt code object, whereas a layout under gpu::precompile_op would pay
                // for a jit compile to produce the same copy.
                if(not std::is_sorted(perm.begin(), perm.end()))
                {
                    auto layout = insert_precompile_op(
                        m, ins, input, make_op("layout", {{"permutation", perm}}));
                    m.replace_instruction(ins, reshape_op, {layout});
                    return;
                }
            }
        }

        auto contiguous = insert_contiguous(m, ins, input);
        m.replace_instruction(ins, reshape_op, {contiguous});
    }
};
} // namespace

void lower_reshape::apply(module& m) const { match::find_matches(m, find_reshape{}); }

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
