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
#ifndef MIGRAPHX_GUARD_GPU_OPTIMIZE_GATHER_HPP
#define MIGRAPHX_GUARD_GPU_OPTIMIZE_GATHER_HPP

#include <migraphx/gpu/config.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

namespace gpu {

/**
 * @brief Pass that annotates gather operations with optimization hints
 * 
 * This pass analyzes gather operations and annotates them with metadata
 * about which optimization strategy should be used. The actual kernel
 * selection happens at compile time in the gather_compiler.
 * 
 * The pass analyzes:
 * - Input/output shapes and sizes
 * - Axis position (innermost vs others)
 * - Memory layout (contiguous vs non-contiguous)
 * - Operation size thresholds
 * 
 * Based on this analysis, it adds hints that the compiler can use to
 * select between basic, optimized, or vectorized gather implementations.
 */
struct MIGRAPHX_GPU_EXPORT optimize_gather
{
    std::string name() const { return "gpu::optimize_gather"; }
    void apply(module& m) const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_OPTIMIZE_GATHER_HPP

