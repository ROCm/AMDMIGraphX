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

// Plugin entry point. Compiled into each MLIR backend plugin DLL together with
// the real mlir.cpp implementation. Exposes a single C factory that returns a
// table of function pointers to the rocMLIR-backed implementations.

#include <migraphx/gpu/mlir.hpp>
#include <migraphx/gpu/mlir_backend.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

extern "C" MIGRAPHX_GPU_EXPORT const mlir_backend_v1* migraphx_gpu_get_mlir_backend()
{
    // Non-capturing lambdas decay to function pointers (the unary '+' forces
    // the conversion and disambiguates the overloaded dump_mlir).
    static const mlir_backend_v1 backend = {
        +[](module m, const std::vector<shape>& inputs) -> std::string {
            return dump_mlir(std::move(m), inputs);
        },
        +[](module m, const std::vector<shape>& inputs, const fs::path& location) {
            dump_mlir_to_file(std::move(m), inputs, location);
        },
        +[](const module& m, const context& migraphx_ctx, const value& solution) -> bool {
            return is_module_fusible(m, migraphx_ctx, solution);
        },
        +[](const context& migraphx_ctx,
            module m,
            const std::vector<shape>& in_shapes,
            const value& solution) -> mlir_code_object {
            return compile_mlir(migraphx_ctx, std::move(m), in_shapes, solution);
        },
        +[](const context& migraphx_ctx,
            module m,
            const std::vector<shape>& inputs,
            bool exhaustive) -> tuning_config {
            return get_tuning_config_mlir(migraphx_ctx, std::move(m), inputs, exhaustive);
        },
        +[](module m, const std::vector<instruction_ref>& inputs, const fs::path& location) {
            dump_mlir_to_mxr(std::move(m), inputs, location);
        },
    };
    return &backend;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
