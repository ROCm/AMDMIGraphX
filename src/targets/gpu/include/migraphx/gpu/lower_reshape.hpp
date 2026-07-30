/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_GPU_LOWER_RESHAPE_HPP
#define MIGRAPHX_GUARD_GPU_LOWER_RESHAPE_HPP

#include <migraphx/gpu/config.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

namespace gpu {

/**
 * Lower `reshape` into a `reshape_lazy` view, inserting a repacking copy only when the
 * input layout does not permit aliasing.
 *
 * A `reshape` preserves row-major logical element order, so it can be a free view over
 * the input buffer whenever the output dims are expressible as a pure restriding of the
 * input. Otherwise the bytes have to be moved first.
 *
 * This pass deliberately runs *after* `eliminate_contiguous`. Lowering inserts a
 * `gpu::contiguous` for many ops and `eliminate_contiguous` then drops the redundant
 * ones globally, so a reshape's real input layout is not settled until that pass has
 * finished. Deciding here means deciding once, with the final layout in hand, instead
 * of emitting a pessimistic copy during lowering and undoing it afterwards.
 *
 * For each `reshape`, in order:
 *
 *  1. If `reshape_lazy` can alias the input directly and lands on exactly the shape the
 *     `reshape` declared, emit just the `reshape_lazy`. No copy.
 *
 *  2. Otherwise, derive the memory order the input would need in order to be aliasable
 *     by running the dim mapping backwards (output dims -> input dims), and verify
 *     forward that repacking into that order does reach the declared shape. If so, emit
 *     a `layout` copy into that order followed by the `reshape_lazy`. The `layout` is
 *     wrapped in `gpu::precompile_op` and compiles to the existing pointwise copy
 *     kernel, so this costs the same single kernel as a standardizing contiguous while
 *     preserving the permutation a following op may want (e.g. NHWC into pooling).
 *
 *  3. Otherwise fall back to a standardizing `gpu::contiguous` followed by the
 *     `reshape_lazy`.
 *
 * Cases 1 and 2 need real strides to reason about, so they are skipped for range-based
 * dynamic inputs, which have none; those always take case 3. Static inputs are lifted
 * to symbolic literals so static and symbolic shapes share one code path.
 *
 * The 2 input form of `reshape` (`reshape(data, output_buffer)`, where the target shape
 * is carried by a runtime-sized output buffer) is rejected. No GPU copy op can express
 * it: they all derive their kernel from an index space shared by source and
 * destination, which a rank-changing copy does not have.
 */
struct MIGRAPHX_GPU_EXPORT lower_reshape
{
    std::string name() const { return "gpu::lower_reshape"; }
    void apply(module& m) const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_LOWER_RESHAPE_HPP
