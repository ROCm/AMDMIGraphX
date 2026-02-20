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
#ifndef MIGRAPHX_GUARD_KERNELS_GATHER_CONCAT_HPP
#define MIGRAPHX_GUARD_KERNELS_GATHER_CONCAT_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/shape.hpp>
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/gather.hpp>

namespace migraphx {

/**
 * Fused gather-concat kernel for 2 gathers
 * 
 * Instead of:
 *   gather(data0, indices0) -> temp0
 *   gather(data1, indices1) -> temp1  
 *   concat(temp0, temp1) -> output
 * 
 * Does:
 *   fused_gather_concat_2(data0, indices0, data1, indices1) -> output
 * 
 * Benefits:
 * - No intermediate tensors (saves memory)
 * - Single kernel launch (reduces overhead)
 * - Better cache locality
 */
template <int GatherAxis, int ConcatAxis, class Input0, class Indices0, 
          class Input1, class Indices1, class Output>
__device__ void gather_concat_2(Input0 input0, Indices0 indices0,
                                Input1 input1, Indices1 indices1,
                                Output output)
{
    auto ind = make_index();
    auto output_shape = output.get_shape();
    auto num_elements = output_shape.elements();
    
    // Get sizes for each gather output
    const auto gather0_size = input0.get_shape().lens[GatherAxis] * indices0.elements();
    const auto gather1_size = input1.get_shape().lens[GatherAxis] * indices1.elements();
    
    ind.global_stride(num_elements, [&](auto i) {
        // Determine which gather segment this element belongs to
        auto concat_axis_size = output_shape.lens[ConcatAxis];
        auto gather0_concat_size = input0.get_shape().lens[ConcatAxis];
        
        // Compute multi-dimensional output index
        auto idx = output_shape.multi(i);
        auto concat_pos = idx[ConcatAxis];
        
        // Determine which gather to use based on concat position
        if(concat_pos < gather0_concat_size)
        {
            // First gather
            auto gather0_idx = idx;
            auto in_index = indices0[gather0_idx[GatherAxis]];
            auto axis_dim_size = input0.get_shape().lens[GatherAxis];
            
            // Normalize negative indices
            in_index = (in_index < 0) ? in_index + axis_dim_size : in_index;
            
            if(__builtin_expect(in_index >= 0 and in_index < axis_dim_size, 1))
            {
                gather0_idx[GatherAxis] = in_index;
                output[i] = input0[gather0_idx];
            }
            else
            {
                MIGRAPHX_ASSERT(false && "Gather out of bounds access");
            }
        }
        else
        {
            // Second gather - adjust concat position
            auto gather1_idx = idx;
            gather1_idx[ConcatAxis] = concat_pos - gather0_concat_size;
            
            auto in_index = indices1[gather1_idx[GatherAxis]];
            auto axis_dim_size = input1.get_shape().lens[GatherAxis];
            
            // Normalize negative indices
            in_index = (in_index < 0) ? in_index + axis_dim_size : in_index;
            
            if(__builtin_expect(in_index >= 0 and in_index < axis_dim_size, 1))
            {
                gather1_idx[GatherAxis] = in_index;
                output[i] = input1[gather1_idx];
            }
            else
            {
                MIGRAPHX_ASSERT(false && "Gather out of bounds access");
            }
        }
    });
}

/**
 * Fused gather-concat kernel for 3 gathers
 */
template <int GatherAxis, int ConcatAxis, 
          class Input0, class Indices0,
          class Input1, class Indices1,
          class Input2, class Indices2,
          class Output>
__device__ void gather_concat_3(Input0 input0, Indices0 indices0,
                                Input1 input1, Indices1 indices1,
                                Input2 input2, Indices2 indices2,
                                Output output)
{
    auto ind = make_index();
    auto output_shape = output.get_shape();
    auto num_elements = output_shape.elements();
    
    // Get concat axis sizes for each gather
    const auto size0 = input0.get_shape().lens[ConcatAxis];
    const auto size1 = input1.get_shape().lens[ConcatAxis];
    const auto size2 = input2.get_shape().lens[ConcatAxis];
    
    ind.global_stride(num_elements, [&](auto i) {
        auto idx = output_shape.multi(i);
        auto concat_pos = idx[ConcatAxis];
        
        if(concat_pos < size0)
        {
            // First gather
            auto gather_idx = idx;
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
            // Second gather
            auto gather_idx = idx;
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
            // Third gather
            auto gather_idx = idx;
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

/**
 * Generic fused gather-concat for N gathers (runtime dispatch)
 * 
 * For more than 3 gathers, use a more flexible approach
 */
template <int GatherAxis, int ConcatAxis, class... Inputs, class Output>
__device__ void gather_concat_n(Output output, Inputs... inputs)
{
    auto ind = make_index();
    auto output_shape = output.get_shape();
    auto num_elements = output_shape.elements();
    
    // Pack inputs into tuple-like structure
    auto input_tuple = pack(inputs...);
    constexpr auto num_gathers = sizeof...(Inputs) / 2;  // data+indices pairs
    
    ind.global_stride(num_elements, [&](auto i) {
        auto idx = output_shape.multi(i);
        auto concat_pos = idx[ConcatAxis];
        
        // Find which gather segment this belongs to
        index_int cumulative_size = 0;
        index_int gather_id = 0;
        
        // Iterate through gathers to find the right one
        for(index_int g = 0; g < num_gathers; ++g)
        {
            auto data_tensor = input_tuple[g * 2];
            auto segment_size = data_tensor.get_shape().lens[ConcatAxis];
            
            if(concat_pos < cumulative_size + segment_size)
            {
                gather_id = g;
                break;
            }
            cumulative_size += segment_size;
        }
        
        // Perform gather for the identified segment
        auto gather_idx = idx;
        gather_idx[ConcatAxis] = concat_pos - cumulative_size;
        
        auto data_tensor = input_tuple[gather_id * 2];
        auto indices_tensor = input_tuple[gather_id * 2 + 1];
        
        auto in_index = indices_tensor[gather_idx[GatherAxis]];
        auto axis_dim_size = data_tensor.get_shape().lens[GatherAxis];
        in_index = (in_index < 0) ? in_index + axis_dim_size : in_index;
        
        if(__builtin_expect(in_index >= 0 and in_index < axis_dim_size, 1))
        {
            gather_idx[GatherAxis] = in_index;
            output[i] = data_tensor[gather_idx];
        }
        else
        {
            MIGRAPHX_ASSERT(false && "Gather out of bounds access");
        }
    });
}

} // namespace migraphx
#endif

