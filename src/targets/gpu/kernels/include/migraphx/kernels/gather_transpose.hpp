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
#ifndef MIGRAPHX_GUARD_KERNELS_GATHER_TRANSPOSE_HPP
#define MIGRAPHX_GUARD_KERNELS_GATHER_TRANSPOSE_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/shape.hpp>
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/gather.hpp>

namespace migraphx {

/**
 * Fused gather+transpose kernel
 * 
 * Instead of:
 *   gather(data, indices) -> temp
 *   transpose(temp) -> output
 * 
 * Does:
 *   fused_gather_transpose(data, indices) -> output (transposed directly)
 * 
 * Benefits:
 * - No intermediate tensor
 * - Single kernel launch
 * - Direct write in transposed layout
 * - Better memory efficiency
 */
template <int GatherAxis, class Permutation, class Input, class Indices, class Output>
__device__ void gather_transpose(Input input, Indices indices, Output output, Permutation perm)
{
    auto ind = make_index();
    auto output_shape = output.get_shape();
    auto num_elements = output_shape.elements();
    auto axis_dim_size = input.get_shape().lens[GatherAxis];
    
    // Create gather output shape (before transpose)
    constexpr auto gather_out_comp = gather_shape<GatherAxis>(
        get_shape_c<Input>{}, get_shape_c<Indices>{});
    
    ind.global_stride(num_elements, [&](auto i) {
        // Get output index (in transposed space)
        auto output_idx = output_shape.multi(i);
        
        // Reverse transpose: map output index back to gather space
        auto gather_idx = output_idx;
        for(index_int d = 0; d < perm.size(); ++d)
        {
            gather_idx[perm[d]] = output_idx[d];
        }
        
        // Perform gather operation
        auto in_index = indices[gather_idx[GatherAxis]];
        
        // Normalize negative indices
        in_index = (in_index < 0) ? in_index + axis_dim_size : in_index;
        
        // Bounds check
        if(__builtin_expect(in_index >= 0 and in_index < axis_dim_size, 1))
        {
            gather_idx[GatherAxis] = in_index;
            output[i] = input[gather_idx];
        }
        else
        {
            MIGRAPHX_ASSERT(false && "Gather out of bounds access");
        }
    });
}

/**
 * Fused gather+transpose for 2 parallel operations feeding into concat
 */
template <int GatherAxis, class Permutation, int ConcatAxis,
          class Input0, class Indices0, class Input1, class Indices1, class Output>
__device__ void gather_transpose_concat_2(Input0 input0, Indices0 indices0,
                                          Input1 input1, Indices1 indices1,
                                          Output output, Permutation perm)
{
    auto ind = make_index();
    auto output_shape = output.get_shape();
    auto num_elements = output_shape.elements();
    
    // Get sizes for each segment (after transpose)
    const auto size0 = input0.get_shape().lens[ConcatAxis];
    
    ind.global_stride(num_elements, [&](auto i) {
        // Get output index (in transposed+concatenated space)
        auto output_idx = output_shape.multi(i);
        
        // Reverse transpose
        auto gather_idx = output_idx;
        for(index_int d = 0; d < perm.size(); ++d)
        {
            gather_idx[perm[d]] = output_idx[d];
        }
        
        auto concat_pos = gather_idx[ConcatAxis];
        
        // Determine which gather segment
        if(concat_pos < size0)
        {
            // First gather
            auto in_index = indices0[gather_idx[GatherAxis]];
            auto axis_dim_size = input0.get_shape().lens[GatherAxis];
            in_index = (in_index < 0) ? in_index + axis_dim_size : in_index;
            
            if(__builtin_expect(in_index >= 0 and in_index < axis_dim_size, 1))
            {
                gather_idx[GatherAxis] = in_index;
                output[i] = input0[gather_idx];
            }
            else
            {
                MIGRAPHX_ASSERT(false && "Gather out of bounds access");
            }
        }
        else
        {
            // Second gather
            gather_idx[ConcatAxis] = concat_pos - size0;
            auto in_index = indices1[gather_idx[GatherAxis]];
            auto axis_dim_size = input1.get_shape().lens[GatherAxis];
            in_index = (in_index < 0) ? in_index + axis_dim_size : in_index;
            
            if(__builtin_expect(in_index >= 0 and in_index < axis_dim_size, 1))
            {
                gather_idx[GatherAxis] = in_index;
                output[i] = input1[gather_idx];
            }
            else
            {
                MIGRAPHX_ASSERT(false && "Gather out of bounds access");
            }
        }
    });
}

/**
 * Fused gather+transpose for 3 parallel operations feeding into concat
 */
template <int GatherAxis, class Permutation, int ConcatAxis,
          class Input0, class Indices0,
          class Input1, class Indices1,
          class Input2, class Indices2,
          class Output>
__device__ void gather_transpose_concat_3(Input0 input0, Indices0 indices0,
                                          Input1 input1, Indices1 indices1,
                                          Input2 input2, Indices2 indices2,
                                          Output output, Permutation perm)
{
    auto ind = make_index();
    auto output_shape = output.get_shape();
    auto num_elements = output_shape.elements();
    
    const auto size0 = input0.get_shape().lens[ConcatAxis];
    const auto size1 = input1.get_shape().lens[ConcatAxis];
    
    ind.global_stride(num_elements, [&](auto i) {
        auto output_idx = output_shape.multi(i);
        
        // Reverse transpose
        auto gather_idx = output_idx;
        for(index_int d = 0; d < perm.size(); ++d)
        {
            gather_idx[perm[d]] = output_idx[d];
        }
        
        auto concat_pos = gather_idx[ConcatAxis];
        
        if(concat_pos < size0)
        {
            auto in_index = indices0[gather_idx[GatherAxis]];
            auto axis_dim_size = input0.get_shape().lens[GatherAxis];
            in_index = (in_index < 0) ? in_index + axis_dim_size : in_index;
            
            if(__builtin_expect(in_index >= 0 and in_index < axis_dim_size, 1))
            {
                gather_idx[GatherAxis] = in_index;
                output[i] = input0[gather_idx];
            }
        }
        else if(concat_pos < size0 + size1)
        {
            gather_idx[ConcatAxis] = concat_pos - size0;
            auto in_index = indices1[gather_idx[GatherAxis]];
            auto axis_dim_size = input1.get_shape().lens[GatherAxis];
            in_index = (in_index < 0) ? in_index + axis_dim_size : in_index;
            
            if(__builtin_expect(in_index >= 0 and in_index < axis_dim_size, 1))
            {
                gather_idx[GatherAxis] = in_index;
                output[i] = input1[gather_idx];
            }
        }
        else
        {
            gather_idx[ConcatAxis] = concat_pos - size0 - size1;
            auto in_index = indices2[gather_idx[GatherAxis]];
            auto axis_dim_size = input2.get_shape().lens[GatherAxis];
            in_index = (in_index < 0) ? in_index + axis_dim_size : in_index;
            
            if(__builtin_expect(in_index >= 0 and in_index < axis_dim_size, 1))
            {
                gather_idx[GatherAxis] = in_index;
                output[i] = input2[gather_idx];
            }
        }
    });
}

} // namespace migraphx
#endif

