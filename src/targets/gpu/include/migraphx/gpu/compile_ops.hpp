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
#ifndef MIGRAPHX_GUARD_GPU_COMPILE_OPS_HPP
#define MIGRAPHX_GUARD_GPU_COMPILE_OPS_HPP

#include <migraphx/gpu/config.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/operation.hpp>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module_pass_manager;

namespace gpu {

struct context;

struct MIGRAPHX_GPU_EXPORT compile_ops
{
    context* ctx         = nullptr;
    bool exhaustive_tune = false;
    bool skip_benchmark  = false;
    /// Save each benchmark candidate here as an MXR file and exit instead of compiling; empty
    /// compiles normally.
    fs::path dump_mxr_dir = {};
    bool keep_recipes     = false;
    std::string name() const { return "gpu::compile_ops"; }
    void apply(module_pass_manager& mpm) const;
};

struct recompile_request
{
    // A gpu::code_object instruction with a recipe
    instruction_ref ins;
    bool exhaustive = false;
};

struct recompiled_kernel
{
    // The new gpu::code_object, with the recipe of the old one and the new solution
    operation code_object;
    // Candidates dropped because they change the kernel signature (for example split-K)
    std::size_t excluded = 0;
    bool benchmarked     = false;
};

/// Recompile gpu::code_object instructions from their recipes for the device of ctx, without
/// changing the program. Only candidates with the same inputs, output and prefills as the
/// original kernel are kept. Throws if a kernel has no such candidate.
MIGRAPHX_GPU_EXPORT std::vector<recompiled_kernel>
recompile_kernels(context& ctx, const std::vector<recompile_request>& requests);

} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_COMPILE_OPS_HPP
