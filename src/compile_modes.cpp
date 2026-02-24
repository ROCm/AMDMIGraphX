/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/compile_modes.hpp>
#include <cstdlib>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

compile_modes convert_to_compile_mode(uint8_t mode)
{
    // If mode is not in range 0-100, return BALANCED
    if(mode > 100)
        return compile_modes::BALANCED;
    
    // Define the enum values as integers for comparison
    constexpr uint8_t eager_val    = static_cast<uint8_t>(compile_modes::EAGER);
    constexpr uint8_t balanced_val = static_cast<uint8_t>(compile_modes::BALANCED);
    constexpr uint8_t max_val      = static_cast<uint8_t>(compile_modes::MAX);
    
    // Calculate distances to each enum value
    uint8_t dist_to_eager    = std::abs(mode - eager_val);
    uint8_t dist_to_balanced = std::abs(mode - balanced_val);
    uint8_t dist_to_max      = std::abs(mode - max_val);
    // Find the minimum distance
    uint8_t min_dist = std::min({dist_to_eager, dist_to_balanced, dist_to_max});
    
    // Return the enum value with minimum distance
    if(min_dist == dist_to_eager)
        return compile_modes::EAGER;
    if(min_dist == dist_to_balanced)
        return compile_modes::BALANCED;
    return compile_modes::MAX;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
