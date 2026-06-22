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
#ifndef MIGRAPHX_GUARD_GPU_LAYOUT_REDUCE_HPP
#define MIGRAPHX_GUARD_GPU_LAYOUT_REDUCE_HPP

#include <migraphx/config.hpp>
#include <migraphx/gpu/export.h>
#include <cstddef>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module_pass_manager;

namespace gpu {

/**
 * Repack the input layout of reductions whose reduced axis is strided so the
 * reduction (and the pointwise ops fused with it, e.g. layernorm) reads
 * coalesced, standard-packed memory instead of a transposed view.
 *
 * This targets the `bmm -> transpose -> layernorm` pattern produced by triangle
 * multiplicative updates: the transpose is a view that leaves the reduced
 * channel axis with a large stride, so reducing over it is uncoalesced. A
 * `layout` op (not `contiguous`, which eliminate_contiguous would strip) is
 * inserted at the view that introduced the strided layout and all of its
 * consumers are rerouted, so the entire normalization runs on the packed
 * buffer. The rewrite only fires once the strided axis is large enough that the
 * coalescing win outweighs the cost of the explicit repack copy.
 */
struct MIGRAPHX_GPU_EXPORT layout_reduce
{
    // Minimum stride (in elements) of the reduced axis for the repack to pay
    // off; below this the copy costs more than the coalescing saves.
    std::size_t min_reduce_stride = std::size_t{1} << 20;
    std::string name() const { return "gpu::layout_reduce"; }
    void apply(module_pass_manager& mpm) const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_LAYOUT_REDUCE_HPP
