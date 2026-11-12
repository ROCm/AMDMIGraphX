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
 *
 */
#ifndef MIGRAPHX_GUARD_GPU_FUSE_GATHER_DOT_HPP
#define MIGRAPHX_GUARD_GPU_FUSE_GATHER_DOT_HPP

#include <migraphx/config.hpp>
#include <migraphx/gpu/export.h>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/optional.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module_pass_manager;

namespace gpu {

// How an instruction reads a tensor through views: a view of the tensor
// itself, or one column of every group of adjacent columns along its last
// axis. The output submodule of a gpu::gather_dot is a pointwise operator
// of such columns of the sums and of its other parameters.
struct gather_dot_column
{
    // the tensor the views read
    instruction_ref sums;
    // the instruction for a view of the tensor itself, else the view that
    // splits the last axis in groups
    instruction_ref view;
    // the slice of the column of every group
    optional<instruction_ref> slice = nullopt;
    std::size_t column              = 0;
    // the columns of a group: one for a view of the tensor itself
    std::size_t group = 1;
};

// The column an instruction reads, if its views keep the order of the
// tensor and nothing else reads them
MIGRAPHX_GPU_EXPORT optional<gather_dot_column> find_gather_dot_column(instruction_ref ins);

// Compute dots whose matrices are gathered per row from a table, eg the
// weights of the experts selected for each token of a mixture of experts
// layer, with one gpu::gather_dot that groups the rows by table entry, so
// each matrix is read once for all the rows gathering it.
//
// The pass replaces the dot and its gathers by a gpu::gather_dot reading the
// table the matrices are computed from, with a weights submodule that
// returns the table, and folds the add of a bias gathered with the same
// indices. It runs before the dots of single rows are rewritten as
// reductions and before the pointwise fusion. fuse_ops then fuses what
// computes that table as the weights submodule and the pointwise operator
// reading the result as the output submodule.
struct MIGRAPHX_GPU_EXPORT fuse_gather_dot
{
    // The number of gathered rows per table entry from which the dots are
    // grouped by entry. With fewer rows the dots of single rows are faster
    // as reductions that read the gathered matrices in place.
    float min_rows_per_entry = 0.5f;

    std::string name() const { return "gpu::fuse_gather_dot"; }
    void apply(module_pass_manager& mpm) const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_FUSE_GATHER_DOT_HPP
