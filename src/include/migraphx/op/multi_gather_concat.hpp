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
#ifndef MIGRAPHX_GUARD_OPERATORS_MULTI_GATHER_CONCAT_HPP
#define MIGRAPHX_GUARD_OPERATORS_MULTI_GATHER_CONCAT_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

// Fused multi-source gather → slice → concat operation
// 
// Pattern:
//   concat[axis=2](
//     slice[k0, axis=0](gather(emb0, idx)),
//     slice[k1, axis=0](gather(emb1, idx)),
//     ...
//   ) → [1, N, sum(embed_dims)]
//
// This kernel takes multiple (embedding, indices) pairs and:
// 1. For each source: gathers from embedding using indices
// 2. Selects specific row(s) from each gather output (via slice)
// 3. Concatenates results along the embedding dimension (axis 2)
//
// Inputs: [emb0, idx0, emb1, idx1, ..., embN, idxN]
// The inputs alternate between embedding tables and their corresponding indices
struct multi_gather_concat
{
    // Per-source metadata:
    // - row_indices[i] = which row to select from gather output for source i
    // - embed_dims[i] = embedding dimension for source i
    // - col_offsets[i] = starting column in output for source i
    std::vector<int64_t> row_indices;    // Which gather row to select per source
    std::vector<int64_t> embed_dims;     // Embedding dimension per source
    std::vector<int64_t> col_offsets;    // Output column offset per source
    int64_t total_embed_dim = 0;         // Sum of all embed_dims
    int64_t num_sources = 0;             // Number of (embedding, indices) pairs

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.row_indices, "row_indices"),
                    f(self.embed_dims, "embed_dims"),
                    f(self.col_offsets, "col_offsets"),
                    f(self.total_embed_dim, "total_embed_dim"),
                    f(self.num_sources, "num_sources"));
    }

    std::string name() const { return "multi_gather_concat"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        // Inputs: [emb0, idx0, emb1, idx1, ...]
        // Each pair contributes to the output
        if(inputs.size() < 2 || inputs.size() % 2 != 0)
            MIGRAPHX_THROW("multi_gather_concat: need even number of inputs (emb, idx pairs)");

        // Get shape info from first indices tensor
        auto idx_shape = inputs[1];
        auto idx_lens = idx_shape.lens();
        
        // Output: [batch, seq_len, total_embed_dim]
        // where batch and seq_len come from indices shape (minus first dim which is sliced)
        std::vector<std::size_t> out_lens;
        out_lens.push_back(1);  // After slice on axis 0
        for(std::size_t i = 1; i < idx_lens.size(); i++)
            out_lens.push_back(idx_lens[i]);
        out_lens.push_back(static_cast<std::size_t>(total_embed_dim));

        return {inputs[0].type(), out_lens};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif

