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
#ifndef MIGRAPHX_GUARD_GPU_RECOMPILE_HPP
#define MIGRAPHX_GUARD_GPU_RECOMPILE_HPP

#include <migraphx/gpu/config.hpp>
#include <migraphx/value.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;
struct module;

namespace gpu {

struct context;

/**
 * The first recompile pass: replace each gpu::code_object kernel of a compiled program with the
 * gpu::precompile_op of its recipe, so compile_ops compiles it again for the device of ctx. The
 * graph, literals, memory plan and schedule are not changed.
 *
 * The device the program was compiled for and its options are read from the comments that
 * record_compiled_for writes with recompilable. Kernels are tuned exhaustively when the recorded
 * or the given options ask for it, see merge_compile_options.
 */
struct MIGRAPHX_GPU_EXPORT restore_recipes
{
    context* ctx = nullptr;
    /// The options recompile is called with, as made by compile_options_value
    value options = {};
    std::string name() const { return "gpu::restore_recipes"; }
    void apply(program& p) const;
};

/// Remove what is left after compile_ops while recompiling: the instructions restore_recipes adds
/// so that compile_ops keeps the instructions of the compiled program that nothing uses, and the
/// precompile ops compile_ops replaced. Dead code elimination cannot be used on a compiled
/// program, since kernels that write into part of a buffer are not used by anything.
struct MIGRAPHX_GPU_EXPORT remove_recompile_leftovers
{
    std::string name() const { return "gpu::remove_recompile_leftovers"; }
    void apply(module& m) const;
};

/// Throw when compile_ops added an allocation while recompiling, since the memory plan of the
/// compiled program has no room for it
struct MIGRAPHX_GPU_EXPORT check_recompiled_memory
{
    std::string name() const { return "gpu::check_recompiled_memory"; }
    void apply(const module& m) const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_RECOMPILE_HPP
