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
#include <migraphx/gpu/layout_reduce.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <algorithm>
#include <numeric>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

namespace {

// View ops that permute/repackage the index space and so can leave their output
// with a non-standard layout. Broadcasts are intentionally excluded: they carry
// zero strides and must not be repacked.
bool is_layout_view(instruction_ref ins)
{
    return contains({"transpose", "reshape", "squeeze", "unsqueeze", "step"}, ins->name());
}

// Matches a reduction over an axis whose stride is large enough that the
// uncoalesced read costs more than repacking the input into a packed buffer.
auto reduces_strided_axis(std::size_t min_stride)
{
    return match::make_basic_pred_matcher([=](instruction_ref ins) {
        const auto s = ins->inputs().front()->get_shape();
        if(s.dynamic() or s.scalar() or s.standard())
            return false;
        const auto v = ins->get_operator().to_value();
        if(not v.contains("axes"))
            return false;
        const auto& lens    = s.lens();
        const auto& strides = s.strides();
        const auto ndim     = static_cast<std::int64_t>(s.ndim());
        const auto axes     = v.at("axes").to_vector<std::int64_t>();
        return std::any_of(axes.begin(), axes.end(), [&](auto axis) {
            axis += axis < 0 ? ndim : 0;
            return axis >= 0 and axis < ndim and lens[axis] > 1 and strides[axis] != 1 and
                   static_cast<std::size_t>(strides[axis]) >= min_stride;
        });
    });
}

// Depth-first walk up the equal-shape, non-broadcast inputs of a strided
// reduction, collecting the view ops that introduced the strided layout.
void collect_layout_sources(instruction_ref ins,
                            const std::vector<std::size_t>& lens,
                            std::vector<instruction_ref>& sources,
                            int depth)
{
    if(depth == 0)
        return;
    for(auto input : ins->inputs())
    {
        const auto s = input->get_shape();
        if(s.dynamic() or s.lens() != lens or contains(s.strides(), 0))
            continue;
        if(is_layout_view(input) and not s.standard())
        {
            if(not contains(sources, input))
                sources.push_back(input);
        }
        else
        {
            collect_layout_sources(input, lens, sources, depth - 1);
        }
    }
}

struct find_strided_reduce
{
    std::size_t min_stride;

    auto matcher() const { return match::reduce(reduces_strided_axis(min_stride)); }

    void apply(module& m, const match::matcher_result& r) const
    {
        const auto reduce = r.result;
        std::vector<instruction_ref> sources;
        collect_layout_sources(reduce, reduce->inputs().front()->get_shape().lens(), sources, 4);
        for(auto src : sources)
        {
            std::vector<std::int64_t> permutation(src->get_shape().ndim());
            std::iota(permutation.begin(), permutation.end(), 0); // identity -> standard packed
            auto packed = m.insert_instruction(
                std::next(src), make_op("layout", {{"permutation", permutation}}), src);
            m.replace_instruction(src, packed);
        }
    }
};

} // namespace

void layout_reduce::apply(module_pass_manager& mpm) const
{
    match::find_matches(mpm.get_module(), find_strided_reduce{min_reduce_stride});
    mpm.run_pass(dead_code_elimination{});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
