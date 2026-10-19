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
#ifndef MIGRAPHX_GUARD_RTGLIB_MIOPEN_WRITE_LITERALS_HPP
#define MIGRAPHX_GUARD_RTGLIB_MIOPEN_WRITE_LITERALS_HPP

#include <migraphx/gpu/config.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
struct module;

namespace gpu {

// Places literals in GPU memory. When they do not fit in max_memory (free GPU memory by default),
// literals that only feed index-selected kernel inputs are left in host memory for the kernel to
// read directly, and the rest are staged through scratch (see staging_peak_percent).
struct MIGRAPHX_GPU_EXPORT write_literals
{
    std::size_t max_memory = 0;
    // Percentage to pad the scratch-memory estimate by. Memory coloring is NP-hard and the
    // liveness-based estimate is incomplete without the scheduler, so the estimate is inflated to
    // avoid exceeding max_memory after compilation.
    std::size_t scratch_overhead_percent = 50;
    // Size cap, as a percentage of the memory shortfall, up to which literals are staged through
    // scratch largest-first (fewest copies); the largest staged copy frees nothing net, so this
    // bounds that waste. Larger literals are staged smallest-first, and only when still needed.
    std::size_t staging_peak_percent = 5;
    std::string name() const { return "gpu::write_literals"; }

    void apply(module& m) const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
