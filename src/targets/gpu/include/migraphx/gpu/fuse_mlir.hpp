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
#ifndef MIGRAPHX_GUARD_GPU_FUSE_MLIR_HPP
#define MIGRAPHX_GUARD_GPU_FUSE_MLIR_HPP

#include <migraphx/gpu/context.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module_pass_manager;

namespace gpu {

MIGRAPHX_GPU_EXPORT bool mlir_enabled();
MIGRAPHX_GPU_EXPORT bool mlir_attention_enabled(context* ctx,
                                                const std::string& use_specific_ops);
MIGRAPHX_GPU_EXPORT bool mlir_flash_decoding_enabled();

struct MIGRAPHX_GPU_EXPORT fuse_mlir
{
    context* ctx = nullptr;
    // Comma-separated list of ops to force on to (or, with a '!'/'~' prefix, off of) MLIR, in the
    // same format as MIGRAPHX_MLIR_USE_SPECIFIC_OPS and supplied via compile_options. Lowest
    // priority: the env var and the architecture and build-config defaults are checked first.
    std::string use_specific_ops = {};
    bool enable_extra            = false;
    std::string name() const { return "gpu::fuse_mlir"; }
    void apply(module_pass_manager& mpm) const;
};

} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_FUSE_MLIR_HPP
