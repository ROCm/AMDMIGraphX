/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/propagate_reshape_layout.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/reshape_dims.hpp>
#include <migraphx/value.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

namespace {
struct find_reshape_lazy_contiguous : match::supports_dynamic_shapes
{
    // eliminate_contiguous only leaves a standardizing gpu::contiguous in front of a
    // reshape_lazy when it could not alias the input directly; that is the only case where a
    // permutation was discarded.
    auto matcher() const
    {
        return match::name("reshape_lazy")(
            match::arg(0)(match::name("gpu::contiguous").bind("contiguous")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto rl       = r.result;
        auto cont     = r.instructions["contiguous"];
        auto input    = cont->inputs().front();
        const auto& s = input->get_shape();
        // A standard input has no permutation to propagate; a range-based dynamic input has
        // no symbolic dims for reshape_dims/find_permutation to work with.
        if(s.standard() or (s.dynamic() and not s.symbolic()))
            return;

        auto sym_in = s.to_symbolic();

        auto permuted = reshape_dims(sym_in, rl->get_shape().sym_dims(), {.lazy = false});
        if(not permuted or permuted->standard())
            return;
        // reshape_dims does not check the element count; bail when it provably differs,
        // matching reshape_lazy::compute_shape (an indeterminate count is allowed through).
        auto out_elems = permuted->sym_elements();
        auto in_elems  = sym_in.sym_elements();
        if(sym::strict_less(out_elems, in_elems).value_or(false) or
           sym::strict_less(in_elems, out_elems).value_or(false))
            return;
        auto relayout = reshape_dims(*permuted, s.sym_dims(), {.lazy = true});
        if(not relayout)
            return;

        auto layout_op    = make_op("layout", {{"permutation", find_permutation(*relayout)}});
        auto layout_shape = layout_op.compute_shape({s});
        auto alloc =
            m.insert_instruction(rl, make_op("allocate", {{"shape", to_value(layout_shape)}}));
        auto layout = m.insert_instruction(
            rl, make_op("gpu::precompile_op", {{"op", to_value(layout_op)}}), input, alloc);
        instruction::replace_argument(rl, cont, layout);
    }
};
} // namespace

void propagate_reshape_layout::apply(module& m) const
{
    match::find_matches(m, find_reshape_lazy_contiguous{});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
