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

#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_BROADCAST_DIMENSIONS_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_BROADCAST_DIMENSIONS_HPP

#include <migraphx/instruction.hpp>
#include <migraphx/common.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

template <typename Builder>
void broadcast_dimensions(Builder& bldr,
                          const std::vector<size_t>& s0_lens,
                          const std::vector<size_t>& s1_lens,
                          const instruction_ref& a0,
                          const instruction_ref& a1,
                          instruction_ref& ba0,
                          instruction_ref& ba1)
{
    // try broadcasting if dimensions other than last two do not match
    if(std::equal(s0_lens.rbegin() + 2, s0_lens.rend(), s1_lens.rbegin() + 2, s1_lens.rend()))
        return;

    auto l0_it = s0_lens.begin() + s0_lens.size() - 2;
    std::vector<std::size_t> l0_broadcasted_lens(s0_lens.begin(), l0_it);
    auto l1_it = s1_lens.begin() + s1_lens.size() - 2;
    std::vector<std::size_t> l1_broadcasted_lens(s1_lens.begin(), l1_it);
    auto output_lens    = compute_broadcasted_lens(l0_broadcasted_lens, l1_broadcasted_lens);
    l0_broadcasted_lens = output_lens;
    l0_broadcasted_lens.insert(l0_broadcasted_lens.end(), l0_it, s0_lens.end());
    l1_broadcasted_lens = output_lens;
    l1_broadcasted_lens.insert(l1_broadcasted_lens.end(), l1_it, s1_lens.end());
    if(s0_lens != l0_broadcasted_lens)
    {
        ba0 = bldr.add_instruction(make_op("multibroadcast", {{"out_lens", l0_broadcasted_lens}}),
                                   a0);
    }
    if(s1_lens != l1_broadcasted_lens)
    {
        ba1 = bldr.add_instruction(make_op("multibroadcast", {{"out_lens", l1_broadcasted_lens}}),
                                   a1);
    }
}

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
