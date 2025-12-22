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
#ifndef MIGRAPHX_GUARD_KERNELS_CONCAT_GATHER_HPP
#define MIGRAPHX_GUARD_KERNELS_CONCAT_GATHER_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/tensor_view.hpp>

namespace migraphx {

// Helper to compute the gather output element given output index
// For gather with axis=0 on embedding table [vocab_size, emb_dim]:
//   output[i, j] = emb[indices[i], j]
template <index_int GatherAxis, class Embedding, class Indices, class OutShape>
__device__ auto gather_element(Embedding emb, Indices indices, OutShape out_shape, index_int flat_idx)
{
    // Convert flat index to multi-index in output shape
    auto out_idx = out_shape.multi(flat_idx);
    
    // Get the index value from indices tensor
    auto idx_pos = out_idx[GatherAxis];
    auto in_index = indices[idx_pos];
    
    // Handle negative indices
    auto axis_dim = emb.get_shape().lens[GatherAxis];
    auto new_in_index = (in_index < 0) ? in_index + axis_dim : in_index;
    
    // Build input index for embedding lookup
    auto emb_idx = out_idx;
    emb_idx[GatherAxis] = new_in_index;
    
    return emb[emb_idx];
}

// Single gather kernel that writes to a slice of the output
// offset: starting position along concat axis
// slice_size: size of this gather's output along concat axis
template <index_int ConcatAxis, index_int GatherAxis, class Embedding, class Indices, class Output>
__device__ void gather_to_slice(index idx, 
                                 Embedding emb, 
                                 Indices indices, 
                                 Output output,
                                 index_int offset,
                                 index_int slice_size)
{
    // Compute shape of gather output
    auto out_lens = output.get_shape().lens;
    out_lens[ConcatAxis] = slice_size;
    auto gather_out_shape = make_shape(out_lens, output.get_shape().strides);
    
    auto gather_elements = indices.get_shape().elements();
    auto emb_trailing_size = emb.get_shape().elements() / emb.get_shape().lens[GatherAxis];
    auto total_elements = gather_elements * emb_trailing_size;
    
    idx.global_stride(total_elements, [&](auto i) {
        // Compute position in gather output
        auto idx_pos = i / emb_trailing_size;
        auto emb_offset = i % emb_trailing_size;
        
        // Get embedding index
        auto in_index = indices[idx_pos];
        auto axis_dim = emb.get_shape().lens[GatherAxis];
        auto new_in_index = (in_index < 0) ? in_index + axis_dim : in_index;
        
        // Read from embedding
        auto emb_val = emb[new_in_index * emb_trailing_size + emb_offset];
        
        // Write to output at offset position
        // For concat axis, we add the offset
        auto out_strides = output.get_shape().strides;
        auto out_idx = idx_pos * out_strides[GatherAxis] + 
                       offset * out_strides[ConcatAxis] + 
                       emb_offset;
        
        output.data()[out_idx] = emb_val;
    });
}

// Replicated gather: single index tensor used N times (multi-hash pattern)
// Equivalent to: gather(emb, concat(idx, idx, ..., idx))  // idx repeated N times
// But avoids creating the intermediate concat tensor
// Output shape: {N, idx_shape..., emb_trailing_dims...}
template <index_int NumReplicas, index_int GatherAxis, class Embedding, class Indices, class Output>
__device__ void replicated_gather(index idx, Embedding emb, Indices indices, Output output)
{
    // For gather[axis=0]: output[r, i, j] = emb[indices[i], j] for each replica r
    auto idx_elements = indices.get_shape().elements();
    auto emb_trailing_size = emb.get_shape().elements() / emb.get_shape().lens[GatherAxis];
    auto elements_per_replica = idx_elements * emb_trailing_size;
    auto total_elements = NumReplicas * elements_per_replica;
    
    idx.global_stride(total_elements, [&](auto i) {
        // Determine which replica and position within replica
        auto replica = i / elements_per_replica;
        auto pos_in_replica = i % elements_per_replica;
        auto idx_pos = pos_in_replica / emb_trailing_size;
        auto emb_offset = pos_in_replica % emb_trailing_size;
        
        // Get embedding index (same for all replicas)
        auto in_index = indices[idx_pos];
        auto axis_dim = emb.get_shape().lens[GatherAxis];
        auto new_in_index = (in_index < 0) ? in_index + axis_dim : in_index;
        
        // Read from embedding
        auto emb_val = emb[new_in_index * emb_trailing_size + emb_offset];
        
        // Write to output: output[replica, idx_pos, emb_offset]
        output.data()[i] = emb_val;
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_CONCAT_GATHER_HPP
