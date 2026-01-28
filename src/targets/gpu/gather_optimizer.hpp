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
#ifndef MIGRAPHX_GUARD_GPU_GATHER_OPTIMIZER_HPP
#define MIGRAPHX_GUARD_GPU_GATHER_OPTIMIZER_HPP

#include <migraphx/shape.hpp>
#include <migraphx/config.hpp>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

/**
 * Enumeration of available gather optimization strategies
 */
enum class gather_optimization
{
    basic,            ///< Basic gather implementation (always works)
    optimized,        ///< Optimized gather with ILP and caching
    vectorized,       ///< Vectorized gather for contiguous patterns
    const_data,       ///< Optimized for constant data with variable indices
    const_data_opt    ///< Constant data with ILP optimization
};

/**
 * Analysis results for gather operation characteristics
 */
struct gather_analysis
{
    std::size_t num_elements;        ///< Total number of output elements
    std::size_t axis_size;           ///< Size of the gather axis
    std::size_t num_indices;         ///< Number of indices to gather
    int axis;                        ///< The gather axis
    bool is_innermost_axis;          ///< True if gathering on innermost dimension
    bool is_contiguous_input;        ///< True if input has standard layout
    bool is_large_gather;            ///< True if output > 10K elements
    bool indices_are_contiguous;     ///< True if indices have standard layout
    bool is_data_constant;           ///< True if data input is constant (@literal or fixed @param)
};

/**
 * Analyzes gather operation characteristics to determine the best optimization
 * 
 * @param inputs Vector of input shapes [data, indices, output]
 * @param axis The gather axis
 * @param data_is_constant Optional hint if data input is known to be constant
 * @return Analysis results
 */
inline gather_analysis analyze_gather(const std::vector<shape>& inputs, 
                                      int axis, 
                                      bool data_is_constant = false)
{
    gather_analysis analysis{};
    
    if(inputs.size() < 3)
        return analysis;
    
    const auto& data_shape = inputs[0];
    const auto& indices_shape = inputs[1];
    const auto& output_shape = inputs[2];
    
    // Basic properties
    analysis.num_elements = output_shape.elements();
    analysis.axis = axis;
    analysis.num_indices = indices_shape.elements();
    analysis.is_data_constant = data_is_constant;
    
    // Check if shapes are standard (contiguous)
    analysis.is_contiguous_input = data_shape.standard();
    analysis.indices_are_contiguous = indices_shape.standard();
    
    // Determine if this is a large gather operation
    constexpr std::size_t large_threshold = 10000;
    analysis.is_large_gather = analysis.num_elements > large_threshold;
    
    // Check if gathering on innermost dimension
    if(!data_shape.dynamic())
    {
        const auto& lens = data_shape.lens();
        analysis.axis_size = lens[axis];
        
        // Innermost axis is the last one for row-major layout
        analysis.is_innermost_axis = (axis == static_cast<int>(lens.size()) - 1);
    }
    
    return analysis;
}

/**
 * Selects the best gather optimization strategy based on operation characteristics
 * 
 * Strategy selection logic:
 * 1. Const Data Optimized: For large constant data gathers (embeddings)
 * 2. Const Data: For medium constant data gathers  
 * 3. Vectorized: When gathering on innermost dimension with contiguous memory
 * 4. Optimized: For medium to large gathers where ILP can be exploited
 * 5. Basic: Fallback for small operations or when other optimizations may not help
 * 
 * @param analysis The gather operation analysis
 * @return The recommended optimization strategy
 */
inline gather_optimization select_gather_optimization(const gather_analysis& analysis)
{
    // Threshold for using optimized vs basic (elements)
    constexpr std::size_t opt_threshold = 1000;
    
    // Threshold for vectorization (elements)
    constexpr std::size_t vec_threshold = 5000;
    
    // Threshold for constant data optimization (elements)
    constexpr std::size_t const_data_threshold = 2000;
    
    // Threshold for constant data with ILP (elements)
    constexpr std::size_t const_data_opt_threshold = 10000;
    
    // Priority 1: Constant data optimizations (common for embeddings/lookups)
    // These work best when:
    // - Data is constant (embedding tables, weight matrices)
    // - Indices are variable (batch processing, sequence inputs)
    // - Access patterns are irregular (not predictable)
    if(analysis.is_data_constant)
    {
        // For very large constant data gathers, use ILP version
        if(analysis.num_elements > const_data_opt_threshold)
        {
            return gather_optimization::const_data_opt;
        }
        
        // For medium constant data gathers, use basic const version
        if(analysis.num_elements > const_data_threshold)
        {
            return gather_optimization::const_data;
        }
        
        // Fall through to standard selection for small constant gathers
    }
    
    // Priority 2: Vectorized optimization for:
    // - Innermost axis gathers (best memory coalescing)
    // - Large operations (> 5K elements)
    // - Contiguous input data
    // - NOT constant data (const data opts are better for that case)
    if(!analysis.is_data_constant &&
       analysis.is_innermost_axis && 
       analysis.num_elements > vec_threshold &&
       analysis.is_contiguous_input)
    {
        return gather_optimization::vectorized;
    }
    
    // Priority 3: Optimized (ILP) version for:
    // - Medium to large operations (> 1K elements)
    // - Not on innermost axis OR not contiguous (vectorized won't help much)
    if(analysis.is_large_gather && analysis.num_elements > opt_threshold)
    {
        return gather_optimization::optimized;
    }
    
    // Default to basic for small operations
    return gather_optimization::basic;
}

/**
 * Converts optimization enum to kernel function name
 */
inline std::string get_gather_kernel_name(gather_optimization opt)
{
    switch(opt)
    {
        case gather_optimization::vectorized:
            return "gather_vectorized";
        case gather_optimization::optimized:
            return "gather_opt";
        case gather_optimization::const_data:
            return "gather_const_data";
        case gather_optimization::const_data_opt:
            return "gather_const_data_opt";
        case gather_optimization::basic:
        default:
            return "gather";
    }
}

/**
 * Determines the optimal gather implementation for given inputs
 * 
 * @param inputs Vector of input shapes [data, indices, output]
 * @param axis The gather axis
 * @param data_is_constant Whether the data input is constant
 * @return String name of the kernel function to use
 */
inline std::string select_gather_kernel(const std::vector<shape>& inputs, 
                                       int axis, 
                                       bool data_is_constant = false)
{
    auto analysis = analyze_gather(inputs, axis, data_is_constant);
    auto optimization = select_gather_optimization(analysis);
    return get_gather_kernel_name(optimization);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif

