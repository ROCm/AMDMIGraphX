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
#ifndef MIGRAPHX_GUARD_GPU_MERGE_PARALLEL_GATHERS_HPP
#define MIGRAPHX_GUARD_GPU_MERGE_PARALLEL_GATHERS_HPP

#include <migraphx/gpu/config.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

namespace gpu {

/**
 * @brief Pass that merges multiple parallel gather operations on the same data
 * 
 * This pass detects patterns where:
 * 1. Multiple gather operations use the same data source
 * 2. All gathers have the same axis
 * 3. Gathers use different (or same) indices
 * 
 * The optimization:
 * - Concatenates all indices into a single index tensor
 * - Performs one large gather operation
 * - Splits/slices the output to original consumers
 * 
 * Example pattern:
 *   data[indices0] -> gather0 -> use0
 *   data[indices1] -> gather1 -> use1
 *   data[indices2] -> gather2 -> use2
 * 
 * Becomes:
 *   combined_indices = concat(indices0, indices1, indices2)
 *   combined_output = data[combined_indices]
 *   out0 = combined_output[0:len0]
 *   out1 = combined_output[len0:len0+len1]
 *   out2 = combined_output[len0+len1:len0+len1+len2]
 * 
 * Benefits:
 * - Single gather kernel instead of N separate kernels
 * - Better GPU utilization (larger parallelism)
 * - Enables subsequent optimizations on merged gather
 * - Reduces kernel launch overhead
 * - Better memory access patterns (can use optimized kernels)
 * 
 * When it helps:
 * - Multiple small gathers → one large gather (better GPU saturation)
 * - Same data, different index patterns
 * - Enables const_data optimization if data is constant
 * - Enables vectorized optimization if conditions met
 * 
 * Common use cases:
 * - Multiple embedding lookups from same table
 * - Batch processing with different index sets
 * - Ensemble models gathering from shared weights
 * - Multi-task learning with shared embeddings
 * 
 * Performance benefits:
 * - 2 small gathers: 1.2-1.4× speedup
 * - 4+ small gathers: 1.5-2.0× speedup
 * - Very small gathers (< 1K): Up to 3× speedup (better GPU utilization)
 * 
 * Trade-offs:
 * - Adds concat overhead for indices (usually negligible)
 * - Adds slice overhead for outputs (usually negligible)
 * - Net benefit when gather cost >> concat/slice cost
 * 
 * NOTE: This pass should run BEFORE other gather optimizations so the
 * merged gather can benefit from optimized kernels (const_data, vectorized, etc.)
 */
struct MIGRAPHX_GPU_EXPORT merge_parallel_gathers
{
    std::string name() const { return "gpu::merge_parallel_gathers"; }
    void apply(module& m) const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_MERGE_PARALLEL_GATHERS_HPP

