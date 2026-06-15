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
#ifndef MIGRAPHX_GUARD_RTGLIB_GPU_MLIR_BACKEND_HPP
#define MIGRAPHX_GUARD_RTGLIB_GPU_MLIR_BACKEND_HPP

// This header defines the ABI used to talk to a dynamically loaded MLIR
// compilation backend plugin (e.g. the legacy rocMLIR backend or the
// rocmlirTriton backend). The plugin exposes a single C entry point that
// returns a table of function pointers wrapping the rocMLIR-touching
// functions declared in mlir.hpp. migraphx_gpu loads exactly one plugin at
// runtime (selected via MIGRAPHX_MLIR_BACKEND) and forwards to it.
//
// This is a proof-of-concept and assumes a shared-libs MIGraphX build, so that
// rich C++ types (module, context, ...) can cross the plugin boundary via the
// single shared migraphx core.

#include <migraphx/gpu/mlir.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
struct module;
namespace gpu {

struct context;

// Version 1 of the MLIR backend vtable. Signatures match the corresponding
// free functions in mlir.hpp exactly.
struct mlir_backend_v1
{
    std::string (*dump_mlir)(module m, const std::vector<shape>& inputs);
    void (*dump_mlir_to_file)(module m,
                              const std::vector<shape>& inputs,
                              const fs::path& location);
    bool (*is_module_fusible)(const module& m, const context& migraphx_ctx, const value& solution);
    mlir_code_object (*compile_mlir)(const context& migraphx_ctx,
                                     module m,
                                     const std::vector<shape>& in_shapes,
                                     const value& solution);
    tuning_config (*get_tuning_config_mlir)(const context& migraphx_ctx,
                                            module m,
                                            const std::vector<shape>& inputs,
                                            bool exhaustive);
    void (*dump_mlir_to_mxr)(module m,
                             const std::vector<instruction_ref>& inputs,
                             const fs::path& location);
};

using mlir_backend_get_fn = const mlir_backend_v1* (*)();

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

// Undecorated name of the exported factory function each plugin must provide.
#define MIGRAPHX_GPU_MLIR_BACKEND_FACTORY_NAME "migraphx_gpu_get_mlir_backend"

#endif // MIGRAPHX_GUARD_RTGLIB_GPU_MLIR_BACKEND_HPP
