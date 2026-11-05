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
#include <migraphx/gpu/device_description.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/value.hpp>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;
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
    bool recompilable     = false;
    std::string name() const { return "gpu::compile_ops"; }
    void apply(module_pass_manager& mpm) const;
};

/// Record the device the program is compiled for with set_compiled_for, so it can be
/// recompiled. Runs last, since passes such as schedule do not expect a comment in the graph.
struct MIGRAPHX_GPU_EXPORT record_compiled_for
{
    context* ctx = nullptr;
    std::string name() const { return "gpu::record_compiled_for"; }
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
    // The new gpu::code_object. The instruction keeps its recipe module.
    operation code_object;
    // The tuning solution the new kernel was compiled with, or null if it has nothing to tune
    value solution;
    // Candidates dropped because they take different arguments or need a fill the program does
    // not have (for example split-K)
    std::size_t excluded = 0;
    bool benchmarked     = false;
};

/// Recompile gpu::code_object instructions from their recipes for the device of ctx, without
/// changing the program. Only candidates with the same inputs and output as the original kernel,
/// and whose fills are already in the program, are kept. Throws if a kernel has no such candidate.
MIGRAPHX_GPU_EXPORT std::vector<recompiled_kernel>
recompile_kernels(context& ctx, const std::vector<recompile_request>& requests);

/// The gpu::precompile_op instruction in the recipe module of a gpu::code_object instruction, or
/// nothing when the kernel was compiled without recompilable
MIGRAPHX_GPU_EXPORT optional<instruction_ref> find_recipe(instruction_ref ins);

/// Give the code objects in root and its submodules that hold the same binary the same
/// module_id, so finalize loads each binary once
MIGRAPHX_GPU_EXPORT void assign_module_ids(module& root);

/// The name of the op a recipe compiles, for example "gpu::mlir_op"
MIGRAPHX_GPU_EXPORT std::string recipe_op_name(instruction_ref recipe);

/// Record the device a recompilable program is compiled for, as a comment instruction with the tag
/// "gpu_target" at the start of m, replacing the one recorded earlier
MIGRAPHX_GPU_EXPORT void set_compiled_for(module& m, const device_description& device);

/// The device recorded with set_compiled_for, or nothing
MIGRAPHX_GPU_EXPORT optional<device_description> get_compiled_for(const module& m);

} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_COMPILE_OPS_HPP
