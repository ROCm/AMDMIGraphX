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
#include <migraphx/gpu/gen/tiling.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/array.hpp>
#include <algorithm>
#include <numeric>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace gen {

static std::size_t integer_divide_ceil(std::size_t x, std::size_t y)
{
    return (x + y - std::size_t{1}) / y;
}

static std::size_t compute_tile_factor(std::size_t r, std::size_t max_size = 64)
{
    std::size_t n = 1;
    auto factors  = make_array(2, 3, 5, 7, 11);
    while(n < max_size)
    {
        auto it = std::find_if(factors.begin(), factors.end(), [&](auto d) { return r % d == 0; });
        if(it == factors.end())
            break;
        r /= *it;
        n *= *it;
    }
    return n;
}

std::size_t compute_vec_size(const std::vector<shape>& inputs)
{
    if(inputs.empty())
        return 1;

    // Default vector sizes to try
    std::vector<std::size_t> sizes = {4, 2};

    // If all inputs are half then only use half2
    if(std::all_of(inputs.begin(), inputs.end(), [](const auto& s) {
           return s.type() == shape::half_type;
       }))
        sizes = {2};

    auto elements = inputs.front().elements();
    for(auto vsize : sizes)
    {
        if(elements % vsize == 0)
        {
            bool all_compatible = std::all_of(inputs.begin(), inputs.end(), [&](const auto& s) {
                return s.elements() % vsize == 0;
            });
            if(all_compatible)
                return vsize;
        }
    }
    return 1;
}

tile_config compute_tile_config(const std::vector<shape>& inputs)
{
    tile_config config;
    if(inputs.empty())
        return config;

    const auto& s = inputs.front();
    auto ndim     = s.ndim();
    auto elements = s.elements();

    config.vec_size = compute_vec_size(inputs);

    // For 1D tensors, no tiling needed
    if(ndim <= 1)
    {
        config.block_size = 256;
        config.grid_size  = integer_divide_ceil(elements / config.vec_size, config.block_size);
        return config;
    }

    // Compute tile dimensions for multi-dimensional tensors
    config.axis = ndim - 2; // Tile along second-to-last dimension

    auto dim1 = compute_tile_factor(s.lens()[config.axis]);
    auto dim2 = compute_tile_factor(s.lens().back(), 4096 / dim1);

    if(dim1 == 1 or dim2 == 1)
    {
        // Fall back to flat tiling
        config.tile_dims.clear();
        config.block_size = 256;
        config.grid_size  = integer_divide_ceil(elements / config.vec_size, config.block_size);
        return config;
    }

    config.tile_dims = {dim1, dim2};
    auto tile_size   = dim1 * dim2;
    config.ntiles    = elements / tile_size;
    config.block_size =
        std::min<std::size_t>(256, integer_divide_ceil(tile_size / config.vec_size, 64) * 64);
    config.grid_size = config.ntiles;

    return config;
}

} // namespace gen
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
