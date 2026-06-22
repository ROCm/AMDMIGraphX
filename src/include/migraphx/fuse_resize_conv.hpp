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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_FUSE_RESIZE_CONV_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_FUSE_RESIZE_CONV_HPP

#include <string>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

// Rewrite `convolution(resize_bilinear_2x(x), w)` (a fixed bilinear-upsample followed by a
// constant-weight 3x3 conv) into a sub-pixel convolution that runs at the LOW (pre-upsample)
// resolution:  interleave_2( convolution(pad(x), w_folded[4K,C,3,3]) ).  The bilinear
// coefficients are folded into the conv weights at compile time, so the upsample is never
// materialized -- the conv reads x once instead of the 4x-larger upsampled tensor.  This is a
// pure memory-traffic win for the bandwidth/latency-bound decoder convs of upscaling models.
struct MIGRAPHX_EXPORT fuse_resize_conv
{
    std::string name() const { return "fuse_resize_conv"; }
    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
