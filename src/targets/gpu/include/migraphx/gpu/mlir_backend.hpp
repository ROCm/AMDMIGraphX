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
// plugins. Inputs are borrowed for each call. Results stay owned by the plugin
// until the host copies their data and calls result_destroy; pointers in result
// views are only valid until that call.

#include <migraphx/filesystem.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/shape.hpp>
#include <cstddef>
#include <cstdint>

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
struct mlir_backend_result;

constexpr std::uint32_t mlir_backend_abi_version = 3;

struct mlir_backend_string_view
{
    const char* data;
    std::size_t size;
};

struct mlir_backend_code_object_view
{
    const std::uint8_t* code_object;
    std::size_t code_object_size;
    mlir_backend_string_view symbol_name;
    std::size_t global;
    std::size_t local;
    const shape* expected_inputs;
    std::size_t expected_input_count;
    const shape* output;
    const std::size_t* prefill_indices;
    const std::int64_t* prefill_values;
    std::size_t prefill_count;
};

struct mlir_backend_tuning_config_view
{
    mlir_backend_string_view problem;
    const mlir_backend_string_view* solutions;
    std::size_t solution_count;
    mlir_backend_string_view detailed_problem_info;
};

struct mlir_backend_v3
{
    std::uint32_t abi_version;
    std::size_t struct_size;

    mlir_backend_result* (*dump_mlir)(
        const module* m, const shape* inputs, std::size_t input_count) noexcept;
    mlir_backend_result* (*dump_mlir_to_file)(const module* m,
                                              const shape* inputs,
                                              std::size_t input_count,
                                              const fs::path::value_type* location,
                                              std::size_t location_size) noexcept;
    mlir_backend_result* (*is_module_fusible)(const module* m,
                                              const context* migraphx_ctx,
                                              const char* solution,
                                              std::size_t solution_size) noexcept;
    mlir_backend_result* (*compile_mlir)(const context* migraphx_ctx,
                                        const module* m,
                                        const shape* in_shapes,
                                        std::size_t in_shape_count,
                                        const char* solution,
                                        std::size_t solution_size) noexcept;
    mlir_backend_result* (*get_tuning_config_mlir)(const context* migraphx_ctx,
                                                   const module* m,
                                                   const shape* inputs,
                                                   std::size_t input_count,
                                                   bool exhaustive) noexcept;
    mlir_backend_result* (*dump_mlir_to_mxr)(const module* m,
                                             const instruction_ref* inputs,
                                             std::size_t input_count,
                                             const fs::path::value_type* location,
                                             std::size_t location_size) noexcept;
    mlir_backend_result* (*mlir_lds_usage_fits_arch)(std::int64_t gemm_o,
                                                     const char* arch,
                                                     std::size_t arch_size,
                                                     shape::type_t elem_type,
                                                     const module* m) noexcept;

    mlir_backend_string_view (*result_error)(const mlir_backend_result* result) noexcept;
    mlir_backend_string_view (*result_string)(const mlir_backend_result* result) noexcept;
    bool (*result_bool)(const mlir_backend_result* result) noexcept;
    mlir_backend_code_object_view (*result_code_object)(
        const mlir_backend_result* result) noexcept;
    mlir_backend_tuning_config_view (*result_tuning_config)(
        const mlir_backend_result* result) noexcept;
    void (*result_destroy)(mlir_backend_result* result) noexcept;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

// Version the exported symbol so older plugins cannot be interpreted as this
// ownership-safe function table.
#define MIGRAPHX_GPU_MLIR_BACKEND_FACTORY_NAME "migraphx_gpu_get_mlir_backend_v3"

#endif // MIGRAPHX_GUARD_GPU_MLIR_BACKEND_HPP
