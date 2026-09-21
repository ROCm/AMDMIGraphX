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

// Export the current rocMLIR-backed implementation through the private plugin
// function table consumed by migraphx_gpu.

#include <migraphx/gpu/mlir.hpp>
#include <migraphx/gpu/mlir_backend.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

extern "C" MIGRAPHX_MLIR_PLUGIN_EXPORT const mlir_backend_v2*
migraphx_gpu_get_mlir_backend_v2()
{
    static const mlir_backend_v2 backend = {
        static_cast<std::string (*)(module, const std::vector<shape>&)>(&dump_mlir),
        &dump_mlir_to_file,
        &is_module_fusible,
        &compile_mlir,
        &get_tuning_config_mlir,
        &dump_mlir_to_mxr,
        &mlir_lds_usage_fits_arch,
    };
    return &backend;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
