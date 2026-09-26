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
#ifndef MIGRAPHX_GUARD_RTGLIB_GPU_MLIR_HPP
#define MIGRAPHX_GUARD_RTGLIB_GPU_MLIR_HPP

#include <chrono>
#include <string>
#include <vector>
#include <migraphx/value.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/gpu/config.hpp>
#include <migraphx/gpu/compile/export.h>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/gpu/tuning_config.hpp>
#include <migraphx/shape.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
struct module;
namespace gpu {

// The MIGRAPHX_GPU_COMPILE_EXPORT functions live in libmigraphx_gpu_compile, which
// migraphx-hiprtc-driver loads without libmigraphx_gpu or the GPU runtime, so they can't use a
// context.

MIGRAPHX_GPU_COMPILE_EXPORT std::string dump_mlir(module m);
MIGRAPHX_GPU_COMPILE_EXPORT std::string dump_mlir(module m, const std::vector<shape>& inputs);
MIGRAPHX_GPU_COMPILE_EXPORT void
dump_mlir_to_file(module m, const std::vector<shape>& inputs, const fs::path& location);

MIGRAPHX_GPU_EXPORT instruction_ref find_final_split(instruction_ref split_ins);

// Throws if the module contains anything other than pointwise, literal, parameter, or return
// instructions.
MIGRAPHX_GPU_EXPORT void validate_pointwise_module(const module& m);

struct MIGRAPHX_GPU_COMPILE_EXPORT mlir_code_object
{
    code_object_op cop;
    std::vector<size_t> prefill_indices = {};
    std::vector<value> prefill_values   = {};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.cop, "cop"),
                    f(self.prefill_indices, "prefill_indices"),
                    f(self.prefill_values, "prefill_values"));
    }
};

// The device properties an MLIR compile reads. It is a plain value so that a compile can run
// where there is no HIP context.
struct MIGRAPHX_GPU_COMPILE_EXPORT mlir_gpu_properties
{
    std::string arch          = "";
    std::size_t cu_count      = 0;
    std::size_t chiplet_count = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.arch, "arch"),
                    f(self.cu_count, "cu_count"),
                    f(self.chiplet_count, "chiplet_count"));
    }
};

MIGRAPHX_GPU_EXPORT mlir_gpu_properties get_mlir_gpu_properties(const context& migraphx_ctx);

MIGRAPHX_GPU_EXPORT bool
is_module_fusible(const module& m, const context& migraphx_ctx, const value& solution);

MIGRAPHX_GPU_COMPILE_EXPORT bool
is_module_fusible(const module& m, const mlir_gpu_properties& props, const value& solution);

// Registers the MLIR dialects and passes and creates the shared thread pool now rather than in
// the first compile
MIGRAPHX_GPU_COMPILE_EXPORT void warm_up_mlir();

// Replace the standard parameters with the actual input layouts and pin the
// output layout, returning the shapes to compile the module with
MIGRAPHX_GPU_COMPILE_EXPORT std::vector<shape>
adjust_param_shapes(module& m, const std::vector<shape>& inputs);

// With a CPU budget, the compile runs in a compile driver session, which gives up once the compile
// has used that much CPU time, and then this throws. It stays in-process when processes are
// disabled on the context or there is no driver.
MIGRAPHX_GPU_EXPORT mlir_code_object
compile_mlir(const context& migraphx_ctx,
             module m,
             const std::vector<shape>& in_shapes,
             const value& solution,
             optional<std::chrono::milliseconds> cpu_budget = nullopt);

MIGRAPHX_GPU_COMPILE_EXPORT mlir_code_object compile_mlir(const mlir_gpu_properties& props,
                                                          module m,
                                                          const std::vector<shape>& in_shapes,
                                                          const value& solution);

MIGRAPHX_GPU_EXPORT instruction_ref insert_mlir(module& m,
                                                instruction_ref ins,
                                                code_object_op co,
                                                const std::vector<instruction_ref>& inputs);

MIGRAPHX_GPU_EXPORT tuning_config get_tuning_config_mlir(const context& migraphx_ctx,
                                                         module m,
                                                         const std::vector<shape>& inputs,
                                                         bool exhaustive);

MIGRAPHX_GPU_COMPILE_EXPORT tuning_config get_tuning_config_mlir(const mlir_gpu_properties& props,
                                                                 module m,
                                                                 const std::vector<shape>& inputs,
                                                                 bool exhaustive);

MIGRAPHX_GPU_COMPILE_EXPORT void
dump_mlir_to_mxr(module m, const std::vector<instruction_ref>& inputs, const fs::path& location);

// Returns true if rocMLIR estimates that a two-gemm problem with
// the given second gemm's output dimension (gemm_o) fits within the target arch's LDS budget
MIGRAPHX_GPU_COMPILE_EXPORT bool mlir_lds_usage_fits_arch(int64_t gemm_o,
                                                          const std::string& arch,
                                                          shape::type_t elem_type,
                                                          const module* m = nullptr);
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
