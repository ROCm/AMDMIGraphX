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
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// Kernel template for fused gather → slice → concat
// This kernel combines:
// 1. gather(embedding, indices) on axis 0
// 2. slice specific rows from gather output
// 3. concat the slices
//
// Output shape: [num_slices, indices.dims[1:], embedding.dims[1:]]
// For output[s, i1, ..., e1, ...]:
//   k = slice_offsets[s]
//   emb_row = indices[k, i1, ...]
//   result = embedding[emb_row, e1, ...]
// NOLINTNEXTLINE
static const char* const gather_slice_concat_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <args.hpp>

namespace migraphx {

// Slice offsets - which rows of gather output to select
__device__ constexpr index_int slice_offsets[] = {${slice_offsets}};
constexpr index_int indices_ndim = ${indices_ndim};
constexpr index_int embedding_ndim = ${embedding_ndim};

extern "C" {

MIGRAPHX_GLOBAL void gather_slice_concat_kernel(void* in_data, void* in_indices, void* output) 
{
    make_tensors()(in_data, in_indices, output)([](auto embedding, auto indices, auto out) {
        auto ind = make_index();
        auto nelem = out.get_shape().elements();
        
        auto out_shape = out.get_shape();
        auto emb_dim0 = embedding.get_shape().lens[0];
        
        ind.global_stride(nelem, [&](auto i) {
            auto out_idx = out_shape.multi(i);
            
            // Output shape: [num_slices, indices.lens[1:], embedding.lens[1:]]
            // out_idx[0] = slice index s → k = slice_offsets[s]
            index_int slice_s = out_idx[0];
            index_int k = slice_offsets[slice_s];
            
            // Build index into indices tensor: [k, out_idx[1], ..., out_idx[indices_ndim-1]]
            // indices has shape [I0, I1, ...], we use [k, out_idx[1:indices_ndim-1]]
            auto idx_pos = indices.get_shape().multi(0);
            idx_pos[0] = k;
            for(index_int d = 1; d < indices_ndim; d++)
            {
                idx_pos[d] = out_idx[d]; // out_idx[1..indices_ndim-1] maps to indices[1..indices_ndim-1]
            }
            
            // Get the embedding row from indices (may be negative)
            auto raw_idx = indices[idx_pos];
            auto emb_row = static_cast<int64_t>(raw_idx);
            
            // Handle negative indices (Python-style)
            if(emb_row < 0) emb_row += static_cast<int64_t>(emb_dim0);
            // Clamp to valid range
            if(emb_row < 0) emb_row = 0;
            if(emb_row >= static_cast<int64_t>(emb_dim0)) emb_row = static_cast<int64_t>(emb_dim0) - 1;
            
            // Build index into embedding: [emb_row, out_idx[indices_ndim], ..., out_idx[out_ndim-1]]
            // Output dims after slice index and indices dims are the embedding dims
            auto emb_pos = embedding.get_shape().multi(0);
            emb_pos[0] = static_cast<index_int>(emb_row);
            for(index_int d = 1; d < embedding_ndim; d++)
            {
                // Embedding dims start at out_idx[1 + (indices_ndim - 1)] = out_idx[indices_ndim]
                emb_pos[d] = out_idx[indices_ndim - 1 + d];
            }
            
            out[i] = embedding[emb_pos];
        });
    });
}

}

} // namespace migraphx

)__migraphx__";

struct gather_slice_concat_compiler : compiler<gather_slice_concat_compiler>
{
    std::vector<std::string> names() const { return {"gather_slice_concat"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        const auto& out_s = inputs.back();
        options.set_launch_params(v, compute_global_for(ctx, out_s.elements()));
        options.inputs         = inputs;
        options.output         = out_s;
        options.virtual_inputs = inputs;
        options.kernel_name    = "gather_slice_concat_kernel";

        // Get slice offsets from operation value
        auto offsets = v.at("slice_offsets").to_vector<int64_t>();
        
        // Build the slice_offsets array string
        std::string offsets_str;
        for(size_t i = 0; i < offsets.size(); i++)
        {
            if(i > 0) offsets_str += ", ";
            offsets_str += std::to_string(offsets[i]);
        }

        // Get dimension info from input shapes
        // inputs: [embedding, indices, output_alloc]
        auto indices_ndim = inputs[1].ndim();
        auto embedding_ndim = inputs[0].ndim();

        auto src = interpolate_string(gather_slice_concat_kernel,
                                      {{"slice_offsets", offsets_str},
                                       {"indices_ndim", std::to_string(indices_ndim)},
                                       {"embedding_ndim", std::to_string(embedding_ndim)}});

        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return compile_op(ctx, to_shapes(ins->inputs()), op.to_value());
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

