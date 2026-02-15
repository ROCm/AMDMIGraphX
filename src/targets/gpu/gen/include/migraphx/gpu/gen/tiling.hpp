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
#ifndef MIGRAPHX_GUARD_GPU_GEN_TILING_HPP
#define MIGRAPHX_GUARD_GPU_GEN_TILING_HPP

#include <migraphx/config.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/gpu/export.h>
#include <vector>
#include <cstddef>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace gen {

struct MIGRAPHX_GPU_EXPORT tile_config
{
    std::vector<std::size_t> tile_dims;
    std::size_t axis       = 0;
    std::size_t block_size = 256;
    std::size_t grid_size  = 1;
    std::size_t vec_size   = 1;
    std::size_t ntiles     = 0;
};

MIGRAPHX_GPU_EXPORT tile_config compute_tile_config(const std::vector<shape>& inputs);

MIGRAPHX_GPU_EXPORT std::size_t compute_vec_size(const std::vector<shape>& inputs);

} // namespace gen
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_GEN_TILING_HPP
