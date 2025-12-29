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
#ifndef MIGRAPHX_GUARD_OPERATORS_GATHER_SLICE_CONCAT_HPP
#define MIGRAPHX_GUARD_OPERATORS_GATHER_SLICE_CONCAT_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

// Fused gather → slice → concat operation
// Takes embedding table and indices, gathers specific rows based on slice_offsets,
// and concatenates them together
struct gather_slice_concat
{
    std::vector<int64_t> slice_offsets; // Which rows to select from gather output
    int64_t concat_axis = 0;            // Axis along which to concatenate
    int64_t gather_axis = 0;            // Axis for gather operation

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.slice_offsets, "slice_offsets"),
                    f(self.concat_axis, "concat_axis"),
                    f(self.gather_axis, "gather_axis"));
    }

    std::string name() const { return "gather_slice_concat"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(2);
        auto data    = inputs[0]; // Embedding table
        auto indices = inputs[1]; // Gather indices

        // Gather output shape: indices.lens + data.lens[1:] = [I0, I1, ..., E1, ...]
        // Slice on axis 0 takes 1 row: [1, I1, ..., E1, ...]
        // Concat num_slices of these: [num_slices, I1, ..., E1, ...]
        auto data_lens    = data.lens();
        auto indices_lens = indices.lens();

        // Output shape: [num_slices, indices_lens[1:], data.lens[1:]]
        std::vector<std::size_t> out_lens;
        out_lens.push_back(slice_offsets.size()); // Number of slices
        // Skip first dim of indices (that's what we're slicing on)
        out_lens.insert(out_lens.end(), indices_lens.begin() + 1, indices_lens.end());
        out_lens.insert(out_lens.end(), data_lens.begin() + 1, data_lens.end());

        return {data.type(), out_lens};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif

