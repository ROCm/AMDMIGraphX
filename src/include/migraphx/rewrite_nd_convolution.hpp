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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_REWRITE_ND_CONVOLUTION_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_REWRITE_ND_CONVOLUTION_HPP

#include <string>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

/**
 * Rewrite a convolution over many spatial dims into one over fewer by folding spatial dims
 * into the channels: the input extent of a folded dim joins the input channels and its output
 * extent the output channels, and the filter gets the tap that connects each pair.
 *
 * Dims where a single input element expands to several outputs are always folded, since each
 * output reads it through one tap and a convolution would compute the others on padding. This
 * undoes the convolutions rewrite_convolution makes with enable_nd_convolution. Other dims are
 * folded, smallest first, until at most max_spatial_dims are left.
 */
struct MIGRAPHX_EXPORT rewrite_nd_convolution
{
    std::size_t max_spatial_dims = 3;
    std::string name() const { return "rewrite_nd_convolution"; }
    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
