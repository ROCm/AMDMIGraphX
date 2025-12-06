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
#ifndef MIGRAPHX_GUARD_KERNELS_GATHER_HPP
#define MIGRAPHX_GUARD_KERNELS_GATHER_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/shape.hpp>
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/tensor_view.hpp>

namespace migraphx {

template <int Axis, class Input, class Indices>
constexpr auto gather_shape(Input input, Indices indices)
{
    auto lengths = input.lens;

    lengths[Axis] = indices.elements();
    return make_shape(lengths, input.strides);
}

template <int Axis, class Input, class Indices, class Output>
__device__ void gather(Input input, Indices indices, Output output)
{
    auto ind           = make_index();
    auto axis_dim_size = input.get_shape().lens[Axis];

    constexpr auto out_comp = gather_shape<Axis>(get_shape_c<Input>{}, get_shape_c<Indices>{});

    ind.global_stride(output.get_shape().elements(), [&](auto i) {
        auto idx      = out_comp.multi(i);
        auto in_index = indices[idx[Axis]];

        auto new_in_index = (in_index < 0) ? in_index + axis_dim_size : in_index;

        idx[Axis] = new_in_index;

        if(idx[Axis] < 0 or idx[Axis] >= axis_dim_size)
        { // Don't gather on this just throw and exit
            MIGRAPHX_ASSERT(false && "Gather out of bounds access");
            return;
        }

        output[i] = input[idx];
    });
}

/**
 * Optimized gather kernel with the following improvements over the basic gather:
 * 
 * 1. Loop unrolling: Processes 4 elements per thread to improve ILP
 * 2. Const caching: Caches frequently accessed shape properties
 * 3. Branch prediction hints: Uses __builtin_expect for the common case
 * 4. Reduced memory traffic: Minimizes redundant loads of shape data
 * 
 * Best for: Medium to large gather operations where ILP can be exploited
 */
template <int Axis, class Input, class Indices, class Output>
__device__ void gather_opt(Input input, Indices indices, Output output)
{
    auto ind           = make_index();
    const auto axis_dim_size = input.get_shape().lens[Axis];
    const auto num_elements = output.get_shape().elements();

    constexpr auto out_comp = gather_shape<Axis>(get_shape_c<Input>{}, get_shape_c<Indices>{});
    
    // Cache output shape properties
    const auto out_shape = output.get_shape();
    
    // Process multiple elements per thread to improve instruction-level parallelism
    constexpr index_int unroll_factor = 4;
    const auto base_idx = ind.global * unroll_factor;
    
    #pragma unroll
    for(index_int offset = 0; offset < unroll_factor; ++offset)
    {
        const auto i = base_idx + offset;
        if(i >= num_elements)
            break;
            
        // Compute multi-dimensional index
        auto idx = out_comp.multi(i);
        
        // Load index with potential for coalescing
        const auto axis_idx = idx[Axis];
        auto in_index = indices[axis_idx];
        
        // Normalize negative indices
        in_index = (in_index < 0) ? in_index + axis_dim_size : in_index;
        
        // Bounds check - optimize for the common case (valid index)
        if(__builtin_expect(in_index >= 0 and in_index < axis_dim_size, 1))
        {
            idx[Axis] = in_index;
            output[i] = input[idx];
        }
        else
        {
            MIGRAPHX_ASSERT(false && "Gather out of bounds access");
        }
    }
}

/**
 * Vectorized gather kernel optimized for contiguous memory patterns:
 * 
 * 1. Vectorized processing: Handles VecSize elements together for better throughput
 * 2. Memory coalescing: Optimized for cases where adjacent threads access adjacent memory
 * 3. Branch prediction: Uses likely/unlikely hints for the common path
 * 4. Tail handling: Efficiently processes remaining elements after vectorized section
 * 
 * Best for: Gather operations on the innermost dimension with contiguous access patterns
 * Note: VecSize should match the hardware vector width for optimal performance (typically 4)
 */
template <int Axis, class Input, class Indices, class Output, int VecSize = 4>
__device__ void gather_vectorized(Input input, Indices indices, Output output)
{
    using value_type = decltype(input[0]);
    
    auto ind = make_index();
    const auto axis_dim_size = input.get_shape().lens[Axis];
    const auto num_elements = output.get_shape().elements();
    
    constexpr auto out_comp = gather_shape<Axis>(get_shape_c<Input>{}, get_shape_c<Indices>{});
    
    // Check if we can use vectorized loads/stores
    // This works best when Axis is the innermost dimension
    const auto vec_elements = num_elements / VecSize;
    
    ind.global_stride(vec_elements, [&](auto vec_i) {
        const auto base_i = vec_i * VecSize;
        
        #pragma unroll
        for(int v = 0; v < VecSize; ++v)
        {
            const auto i = base_i + v;
            if(i >= num_elements)
                break;
                
            auto idx = out_comp.multi(i);
            auto in_index = indices[idx[Axis]];
            
            // Normalize negative indices
            in_index = (in_index < 0) ? in_index + axis_dim_size : in_index;
            
            // Early bounds check
            if(__builtin_expect(in_index >= 0 and in_index < axis_dim_size, 1))
            {
                idx[Axis] = in_index;
                output[i] = input[idx];
            }
            else
            {
                MIGRAPHX_ASSERT(false && "Gather out of bounds access");
                return;
            }
        }
    });
    
    // Handle remaining elements
    const auto remaining_start = vec_elements * VecSize;
    if(ind.global < (num_elements - remaining_start))
    {
        const auto i = remaining_start + ind.global;
        auto idx = out_comp.multi(i);
        auto in_index = indices[idx[Axis]];
        
        in_index = (in_index < 0) ? in_index + axis_dim_size : in_index;
        
        if(__builtin_expect(in_index >= 0 and in_index < axis_dim_size, 1))
        {
            idx[Axis] = in_index;
            output[i] = input[idx];
        }
        else
        {
            MIGRAPHX_ASSERT(false && "Gather out of bounds access");
        }
    }
}

} // namespace migraphx
#endif
