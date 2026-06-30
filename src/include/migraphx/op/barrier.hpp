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
#ifndef MIGRAPHX_GUARD_OPERATORS_BARRIER_HPP
#define MIGRAPHX_GUARD_OPERATORS_BARRIER_HPP

#include <migraphx/argument.hpp>
#include <migraphx/check_shapes.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

// A tagged pass-through with the same value semantics as identity. It blocks
// optimizations from rewriting across the point it marks (for example the fp16
// round-trip convert folding and precision propagation that fast_mm needs to
// preserve). The `tag` lets a specific kind of barrier be targeted; the
// eliminate_barrier pass removes barriers once they are no longer needed.
struct barrier
{
    std::string tag = "";

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.tag, "tag"));
    }

    std::string name() const { return "barrier"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        return inputs.at(0);
    }

    argument compute(shape, std::vector<argument> args) const { return args[0]; }

    value attributes() const { return {{"pointwise", true}, {"point_op", "${0}"}}; }

    std::vector<std::size_t> output_alias(const std::vector<shape>&) const { return {0}; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
