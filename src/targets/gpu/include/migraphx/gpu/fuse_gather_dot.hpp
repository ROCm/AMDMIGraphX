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
#include <migraphx/module_ref.hpp>
#include <migraphx/optional.hpp>
#include <string>
#include <vector>

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

// Where the inputs of a gpu::gather_dot are: the rows, the indices, the
// weight tensors, the bias if there is one, then the other inputs of the
// output submodule. The output buffer follows them after lowering.
struct gather_dot_layout
{
    static constexpr std::size_t rows         = 0;
    static constexpr std::size_t indices      = 1;
    static constexpr std::size_t first_weight = 2;

    std::size_t weights = 1;
    bool bias           = false;
    std::size_t others  = 0;

    std::size_t bias_index() const { return first_weight + weights; }
    std::size_t first_other() const { return bias_index() + (bias ? 1 : 0); }
    // the number of inputs without the output buffer
    std::size_t size() const { return first_other() + others; }
};

// The layout of ninputs inputs, not counting the output buffer: the
// submodules give the weights and the other inputs, and an input left over
// is the bias
MIGRAPHX_GPU_EXPORT gather_dot_layout get_gather_dot_layout(std::size_t ninputs,
                                                            const std::vector<module_ref>& mods);

// What the kernel needs from the weights submodule
struct gather_dot_weights
{
    // the weight tensors that store two int4 elements per byte
    std::vector<std::size_t> packed = {};
    // the axis they are unpacked along
    std::size_t unpack_axis = 0;
    // the pointwise submodule computing an element of the matrices, or
    // nothing when the weight tensor is the table
    const_module_ref function = nullptr;
};

// The weights submodule returns its parameter, which is the table, or a
// pointwise operator of its parameters in their order, of which some are
// read through an unpack_int4. Throws for any other submodule.
MIGRAPHX_GPU_EXPORT gather_dot_weights read_gather_dot_weights(const module& m);

// What the kernel needs from the output submodule
struct gather_dot_output
{
    // the columns of a group
    std::size_t group = 1;
    // the column of the group each input of the pointwise operator reads,
    // nothing for the next of the other inputs
    std::vector<optional<std::size_t>> args = {};
    // the pointwise submodule computing an element of the result
    const_module_ref function = nullptr;
};

// The output submodule returns a pointwise operator of columns of the sums,
// its first parameter, and of its other parameters in their order. Throws
// for any other submodule.
MIGRAPHX_GPU_EXPORT gather_dot_output read_gather_dot_output(const module& m);

// Compute dots whose matrices are gathered per row from a table, eg the
// weights of the experts selected for each token of a mixture of experts
// layer, with one gpu::gather_dot that groups the rows by table entry, so
// each matrix is read once for all the rows gathering it.
//
// The pass replaces the dot and its gathers by a gpu::gather_dot reading the
// table of matrices, and folds the add of a bias gathered with the same
// indices. It runs before the dots of single rows are rewritten as
// reductions and before the pointwise fusion; fuse_ops then fuses what
// computes the table and the pointwise operator reading the result.
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
