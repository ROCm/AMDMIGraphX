/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
 *
 */
#ifndef MIGRAPHX_GUARD_MIGRAPHX_SPLIT_REDUCE_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_SPLIT_REDUCE_HPP

#include <migraphx/config.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module_pass_manager;

/// This pass will split large fused_reduce operators so that the reduction
/// will happen across multiple compute units gaining better occupancy for
/// targets with many compute units. For reductions larger than the
/// lower_split_size, the reduce axis is split into groups by reshaping the
/// inputs(so {M, N} becomes {M, G, N/G}), a first fused_reduce computes a
/// partial reduction for each group, and the trailing module completes it
/// with another reduction over the groups. For reductions larger than the
/// split_size, the atomic-based split_fused_reduce can be used instead,
/// which splits any elementwise operators into separate operators as well
/// due to needing global synchronization. When both thresholds are
/// applicable, prefer_partial_reduce selects which one is used.
struct MIGRAPHX_EXPORT split_reduce
{
    /// Threshold to use the atomic-based split_fused_reduce
    std::size_t split_size = 8192;
    /// Threshold to split into a partial reduction that is completed by a
    /// second fused_reduce, when the batch is below lower_max_batch
    std::size_t lower_split_size = 8192;
    /// Threshold where the reduction is too large for a single workgroup:
    /// beyond this the resident rows overflow the last-level cache so the
    /// fused kernel can no longer re-read its row from cache(and the
    /// register limits force the block_large fallback), so the partial
    /// reduction is used regardless of the batch
    std::size_t upper_split_size = 524288;
    /// For reductions below the upper_split_size, only split when the
    /// batch(the number of reduction outputs) is below this, since with one
    /// workgroup per output a large batch already has enough parallelism
    /// and splitting it would only add another read of the input
    std::size_t lower_max_batch = 64;
    /// Use the partial reduction when both thresholds are applicable
    bool prefer_partial_reduce = true;
    std::string name() const { return "split_reduce"; }
    void apply(module_pass_manager& mpm) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_SPLIT_REDUCE_HPP
