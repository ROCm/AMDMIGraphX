/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_OP_LAYOUT_HPP
#define MIGRAPHX_GUARD_OP_LAYOUT_HPP

#include <migraphx/config.hpp>
#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/op/unary.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 * Rearrange the memory layout of the input instruction based on the permutation attribute.
 * This operator changes the order of elements in memory, *not* the order in the tensor.
 * Therefore, regardless of how the memory layout is changed, the order of elements returned by a
 * tensor_view will be unchanged.
 * `permutation`: List with how to rearrange the data buffer of the input instruction. This
 * permutation is the transpose from the order in the tensor to the order in memory.
 */
struct layout : unary<layout>
{
    std::vector<int64_t> permutation;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.permutation, "permutation"));
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1).only_dims(permutation.size());
        auto lens = inputs.at(0).lens();
        auto t    = inputs.at(0).type();
        return shape::from_permutation(t, lens, permutation);
    }

    auto apply() const
    {
        return [](auto x) { return x; };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_OP_LAYOUT_HPP
