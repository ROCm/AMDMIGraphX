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
#ifndef MIGRAPHX_GUARD_RTGLIB_REWRITE_CONVOLUTION_HPP
#define MIGRAPHX_GUARD_RTGLIB_REWRITE_CONVOLUTION_HPP

#include <string>
#include <migraphx/config.hpp>
#include <migraphx/instruction_ref.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

/**
 * Rewrite convolution_backwards (transposed convolution) into a set of stride-1 forward
 * convolutions plus reassembly, following MIOpen's implicit-gemm backward-data v4r1 algorithm.
 *
 * A backward-data convolution with stride S and dilation D is split, per spatial dim, into
 * Ytilda = S / gcd(S, D) residue "gemms". Each residue is a dense stride-1 forward convolution on
 * a strided slice of the (flipped) filter; the residue outputs are interleaved back into the
 * spatial grid. This lets downstream fusion (e.g. MLIR) generate several small, efficient kernels
 * instead of one monolithic backward convolution.
 */
struct MIGRAPHX_EXPORT rewrite_convolution
{
    std::string name() const { return "rewrite_convolution"; }
    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
