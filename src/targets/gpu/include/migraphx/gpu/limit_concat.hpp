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
#ifndef MIGRAPHX_GUARD_GPU_LIMIT_CONCAT_HPP
#define MIGRAPHX_GUARD_GPU_LIMIT_CONCAT_HPP

#include <migraphx/config.hpp>
#include <migraphx/gpu/export.h>
#include <cstddef>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

namespace gpu {

// fuse_concat flattens a same-axis chain into one concat, and fuse_ops can
// still add trailing kernel arguments after that. hiprtc aborts on the
// resulting kernel (the CMZ segmentation unroll is ~592 inputs: exit
// 0xC0000409, reported as a concat with no tuning config). This pass runs
// after fuse_ops and splits any wider concat into kernels of at most
// max_inputs. 256 is the low end of the range that still compiles.
struct MIGRAPHX_GPU_EXPORT limit_concat_inputs
{
    std::size_t max_inputs = 256;

    std::string name() const { return "gpu::limit_concat_inputs"; }
    void apply(module& m) const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_LIMIT_CONCAT_HPP
