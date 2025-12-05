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
#ifndef MIGRAPHX_GUARD_SPLIT_FACTOR_HPP
#define MIGRAPHX_GUARD_SPLIT_FACTOR_HPP

#include <algorithm>
#include <cstddef>
#include <migraphx/array.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/**
 * Calculate split factor for a dimension to make it less than min_size.
 *
 * This function finds the largest divisor that can divide the dimension
 * to make it less than min_size. It uses prime factors [2, 3, 5, 7, 11]
 * to find good divisors that work well for parallel execution.
 *
 * Used by:
 * - rewrite_topk: Splits large topk operations for better performance
 * - flash_decoding: Splits attention sequence dimension for parallelization
 * - split_reduce (GPU JIT): Splits reduction operations
 *
 * @param r The dimension size to split (will be modified to remaining size)
 * @param min_size The minimum size threshold
 * @return The split factor (number of groups)
 */
inline std::size_t split_dim(std::size_t& r, std::size_t min_size)
{
    std::size_t n = 1;
    auto factors  = make_array(2, 3, 5, 7, 11);
    while(r > min_size)
    {
        // NOLINTNEXTLINE(readability-qualified-auto)
        auto it = std::find_if(factors.begin(), factors.end(), [&](auto d) { return r % d == 0; });
        if(it == factors.end())
            break;
        r /= *it;
        n *= *it;
    }
    return n;
}

/**
 * Calculate split factor with maximum splits constraint.
 *
 * Similar to split_dim but also respects a maximum number of splits.
 * Useful when there's a limit on parallelization due to hardware constraints.
 *
 * @param dimension The dimension size to split
 * @param min_size Minimum size per chunk after splitting
 * @param max_splits Maximum number of splits allowed (0 = no limit)
 * @return The split factor that respects both constraints
 */
inline std::size_t
split_dim_with_max(std::size_t dimension, std::size_t min_size, std::size_t max_splits = 0)
{
    // Make a copy since split_dim modifies the value
    std::size_t remaining  = dimension;
    std::size_t num_splits = split_dim(remaining, min_size);

    // If no max constraint or already within limit, return as is
    if(max_splits == 0 || num_splits <= max_splits)
        return num_splits;

    // Reduce splits to respect max_splits constraint
    auto factors = make_array(2, 3, 5, 7, 11);
    while(num_splits > max_splits)
    {
        // Remove the smallest prime factor to reduce splits
        for(auto factor : factors)
        {
            if(num_splits % factor == 0)
            {
                num_splits /= factor;
                remaining *= factor;
                break;
            }
        }
        // Safety check to avoid infinite loop
        if(num_splits > max_splits && num_splits < 2)
            break;
    }

    return num_splits;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_SPLIT_FACTOR_HPP
