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
#ifndef MIGRAPHX_GUARD_GPU_GEN_TILING_HPP
#define MIGRAPHX_GUARD_GPU_GEN_TILING_HPP

#include <migraphx/config.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/gpu/gen/export.h>
#include <vector>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

namespace gpu {
namespace gen {

/// Tiling configuration computed from input shapes
struct MIGRAPHX_GPU_GEN_EXPORT tile_config
{
    std::vector<std::size_t> tile_dims = {};
    std::size_t axis                   = 0;
    std::size_t ntiles                 = 0;
    std::size_t block_size             = 0;
    std::vector<std::size_t> inner     = {};
    std::vector<std::size_t> outer     = {};

    bool is_tiled() const { return ntiles > 0; }

    /// Compute tile configuration from input shapes
    static tile_config compute(const std::vector<shape>& inputs, std::size_t noutputs);
};

/// Pass that applies tiling transformations to gen IR modules
struct MIGRAPHX_GPU_GEN_EXPORT gen_tiling
{
    std::string name() const { return "gpu::gen::tiling"; }
    void apply(module& m) const;
};

} // namespace gen
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_GEN_TILING_HPP
