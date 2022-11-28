/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/rewrite_ops.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/permutation.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace {

MIGRAPHX_PRED_MATCHER(col_matrix, instruction_ref ins)
{
    if(not ins->get_shape().transposed())
        return false;
    if(ins->get_shape().ndim() < 2)
        return false;
    auto perm = find_permutation(ins->get_shape());
    auto n    = perm.size() - 1;
    return perm[n] == n - 1 and perm[n - 1] == n;
}

MIGRAPHX_PRED_MATCHER(broadcast_matrix_dims, instruction_ref ins)
{
    if(not ins->get_shape().broadcasted())
        return false;
    if(ins->get_shape().ndim() < 2)
        return false;
    return std::any_of(ins->get_shape().lens().rbegin(),
                       ins->get_shape().lens().rend() + 2,
                       [](auto i) { return i == 0; });
}

struct find_dot_const
{
    auto matcher() const
    {
        return match::name("dot")(match::arg(1)(
            match::is_constant(),
            match::none_of(col_matrix(), broadcast_matrix_dims()),
            match::skip_broadcasts(match::any().bind("w"))))(match::none_of(match::is_constant()));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins = r.result;
        auto w   = r.instructions["w"];
        if(w->get_shape().ndim() < 2)
            return;
        auto perm = find_permutation(w->get_shape());
        auto n    = perm.size() - 1;
        std::swap(perm[n], perm[n - 1]);
        auto wl = m.insert_instruction(std::next(w), make_op("layout", {{"permutation", perm}}), w);
        m.replace_instruction(w, wl);
    }
};

} // namespace

void rewrite_ops::apply(module& m) const { match::find_matches(m, find_dot_const{}); }

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
