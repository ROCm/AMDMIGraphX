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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_FUSE_REDUCE_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_FUSE_REDUCE_HPP

#include <migraphx/config.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/module_ref.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module_pass_manager;

struct MIGRAPHX_EXPORT fuse_reduce
{
    std::string name() const { return "fuse_reduce"; }
    void apply(module_pass_manager& mpm) const;

    bool enable_rewrite_reshapes   = true;
    bool enable_rewrite_broadcasts = false;
    // Move the gathers along non-reduced axes into the reduce. A reduce with
    // gathered inputs can no longer be remapped by the reshape and broadcast
    // rewrites, so this is left to the last reduce fusion of a pipeline.
    bool enable_gather = true;
};

/// Inserts the submodule of ins into sm with its parameters mapped to the inputs of ins
MIGRAPHX_EXPORT std::vector<instruction_ref>
insert_module_in_submodule(module_ref sm,
                           instruction_ref ins,
                           std::unordered_map<instruction_ref, instruction_ref>* map_ins = nullptr);

/// Removes the duplicate and dead instructions left by fusing into a reduce module
MIGRAPHX_EXPORT void finalize_reduce_module(module_ref m);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_FUSE_POINTWISE_HPP
