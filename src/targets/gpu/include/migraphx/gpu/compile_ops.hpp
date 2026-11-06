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
#include <migraphx/optional.hpp>
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
    /// Keep each kernel's precompile op in a recipe submodule, so it can be recompiled
    bool recompilable = false;
    /// Also tune exhaustively when the context asks for it. Recompile sets the context from the
    /// options recorded with the program, which are only known once the passes run.
    bool exhaustive_from_context = false;
    std::string name() const { return "gpu::compile_ops"; }
    void apply(module_pass_manager& mpm) const;
};

/// The gpu::precompile_op instruction in the recipe module of a gpu::code_object instruction, or
/// nothing when the kernel was compiled without recompilable
MIGRAPHX_GPU_EXPORT optional<instruction_ref> find_recipe(instruction_ref ins);

/// The inputs the precompile op in the recipe of a gpu::code_object instruction had, found from
/// the kernel, or nothing when the kernel has no recipe or a recipe from an older build
MIGRAPHX_GPU_EXPORT optional<std::vector<instruction_ref>> recipe_inputs(instruction_ref ins);

/// The name of the op a recipe compiles, for example "gpu::mlir_op"
MIGRAPHX_GPU_EXPORT std::string recipe_op_name(instruction_ref recipe);

} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_COMPILE_OPS_HPP
