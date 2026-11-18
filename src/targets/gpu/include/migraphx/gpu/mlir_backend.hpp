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
#ifndef MIGRAPHX_GUARD_GPU_MLIR_BACKEND_HPP
#define MIGRAPHX_GUARD_GPU_MLIR_BACKEND_HPP

// Private same-build interface between migraphx_gpu and its MLIR backend
// plugins. Version 2 adds the LDS-usage query required by the 2611 MLIR fusion
// pipeline. Rich C++ types may cross this boundary only because the plugins are
// built and shipped with the matching shared MIGraphX build.

#include <migraphx/gpu/mlir.hpp>

#ifdef _WIN32
#define MIGRAPHX_MLIR_PLUGIN_EXPORT __declspec(dllexport)
#else
#define MIGRAPHX_MLIR_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
struct module;
namespace gpu {

struct context;

struct mlir_backend_v2
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
    bool (*mlir_lds_usage_fits_arch)(
        int64_t gemm_o, const std::string& arch, shape::type_t elem_type, const module* m);
};

using mlir_backend_get_fn = const mlir_backend_v2* (*)();

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

// Version the exported symbol so a plugin from the original POC cannot be
// interpreted as the larger 2611 function table.
#define MIGRAPHX_GPU_MLIR_BACKEND_FACTORY_NAME "migraphx_gpu_get_mlir_backend_v2"

#endif // MIGRAPHX_GUARD_GPU_MLIR_BACKEND_HPP
