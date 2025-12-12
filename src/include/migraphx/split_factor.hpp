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
#include <limits>
#include <migraphx/array.hpp>
#include <iostream>
namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/**
 * Calculate split factor with or without maximum splits constraint.
 *
 * Used by:
 * - rewrite_topk: Splits large topk operations for better performance
 * - flash_decoding: Splits attention sequence dimension for parallelization
 * - split_reduce (GPU JIT): Splits reduction operations
 *
 * To compute the number of split groups it finds the largest
 * divisor that can divide dimension to make it less than min_size.
 *
 * @param r The value to split. This is passed by reference and will be modified to the remaining value after splitting.
 * @param min_size Target threshold - splits until remaining size is less than this value
 * @param max_splits Target threshold - if reached, returns the smallest split factor greater than
 * or equal to max_splits that evenly divides dimension. Optional
 * @return The split factor that respects both constraints
 */

inline std::size_t split_dim(std::size_t& r,
                             std::size_t min_size,
                             std::size_t max_splits = std::numeric_limits<std::size_t>::max())
{
    std::size_t n = 1;
    auto factors  = make_array(2, 3, 5, 7, 11);
    while(r > min_size and n < max_splits)
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

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_SPLIT_FACTOR_HPP
