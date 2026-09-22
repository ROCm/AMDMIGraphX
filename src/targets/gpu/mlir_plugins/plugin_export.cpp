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
#include <migraphx/errors.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/module.hpp>
#include <algorithm>
#include <cassert>
#include <exception>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct mlir_backend_result
{
    std::string data;
    std::string error;
    std::unique_ptr<mlir_code_object> code_object;
    std::vector<std::int64_t> prefill_values;
    std::string tuning_problem;
    std::vector<std::string> tuning_solutions;
    std::vector<mlir_backend_string_view> tuning_solution_views;
    std::string detailed_problem_info;
    bool bool_value = false;
};

static mlir_backend_result* make_error_result(const char* message) noexcept
{
    try
    {
        auto result   = std::make_unique<mlir_backend_result>();
        result->error =
            message == nullptr or *message == '\0' ? "MLIR backend plugin error" : message;
        return result.release();
    }
    catch(...)
    {
        return nullptr;
    }
}

template <class F>
static mlir_backend_result* make_backend_result(F f) noexcept
{
    try
    {
        auto result = std::make_unique<mlir_backend_result>();
        f(*result);
        return result.release();
    }
    catch(const std::exception& e)
    {
        return make_error_result(e.what());
    }
    catch(...)
    {
        return make_error_result("Unknown MLIR backend plugin error");
    }
}

template <class T>
static const T& required(const T* x, const char* name)
{
    if(x == nullptr)
        MIGRAPHX_THROW(std::string{"Missing MLIR backend argument: "} + name);
    return *x;
}

template <class T>
static std::vector<T> copy_range(const T* data, std::size_t size)
{
    if(size == 0)
        return {};
    if(data == nullptr)
        MIGRAPHX_THROW("Missing MLIR backend range");
    return {data, data + size};
}

static std::string copy_string(const char* data, std::size_t size)
{
    if(size == 0)
        return {};
    if(data == nullptr)
        MIGRAPHX_THROW("Missing MLIR backend string");
    return {data, size};
}

static fs::path copy_path(const fs::path::value_type* data, std::size_t size)
{
    if(size == 0)
        return {};
    if(data == nullptr)
        MIGRAPHX_THROW("Missing MLIR backend path");
    return fs::path{fs::path::string_type{data, size}};
}

static mlir_backend_string_view make_view(const std::string& x) noexcept
{
    return {x.data(), x.size()};
}

static std::int64_t prefill_value(const value& v)
{
    const auto* result = v.if_int64();
    if(result == nullptr)
        MIGRAPHX_THROW("MLIR prefill value must be an integer");
    return *result;
}

static mlir_backend_result*
plugin_dump_mlir(const module* m, const shape* inputs, std::size_t input_count) noexcept
{
    return make_backend_result([&](auto& result) {
        result.data = dump_mlir(required(m, "module"), copy_range(inputs, input_count));
    });
}

static mlir_backend_result* plugin_dump_mlir_to_file(const module* m,
                                                     const shape* inputs,
                                                     std::size_t input_count,
                                                     const fs::path::value_type* location,
                                                     std::size_t location_size) noexcept
{
    return make_backend_result([&](auto&) {
        dump_mlir_to_file(required(m, "module"),
                          copy_range(inputs, input_count),
                          copy_path(location, location_size));
    });
}

static mlir_backend_result* plugin_is_module_fusible(const module* m,
                                                     const context* migraphx_ctx,
                                                     const char* solution,
                                                     std::size_t solution_size) noexcept
{
    return make_backend_result([&](auto& result) {
        result.bool_value =
            is_module_fusible(required(m, "module"),
                              required(migraphx_ctx, "context"),
                              value(copy_string(solution, solution_size)));
    });
}

static mlir_backend_result* plugin_compile_mlir(const context* migraphx_ctx,
                                                const module* m,
                                                const shape* in_shapes,
                                                std::size_t in_shape_count,
                                                const char* solution,
                                                std::size_t solution_size) noexcept
{
    return make_backend_result([&](auto& result) {
        auto object = compile_mlir(required(migraphx_ctx, "context"),
                                   required(m, "module"),
                                   copy_range(in_shapes, in_shape_count),
                                   value(copy_string(solution, solution_size)));
        result.prefill_values.reserve(object.prefill_values.size());
        std::transform(object.prefill_values.begin(),
                       object.prefill_values.end(),
                       std::back_inserter(result.prefill_values),
                       &prefill_value);
        if(result.prefill_values.size() != object.prefill_indices.size())
            MIGRAPHX_THROW("Mismatched MLIR prefill metadata");
        object.prefill_values.clear();
        result.code_object = std::make_unique<mlir_code_object>(std::move(object));
    });
}

static mlir_backend_result* plugin_get_tuning_config_mlir(const context* migraphx_ctx,
                                                          const module* m,
                                                          const shape* inputs,
                                                          std::size_t input_count,
                                                          bool exhaustive) noexcept
{
    return make_backend_result([&](auto& result) {
        auto config = get_tuning_config_mlir(required(migraphx_ctx, "context"),
                                             required(m, "module"),
                                             copy_range(inputs, input_count),
                                             exhaustive);
        const auto* problem = config.problem.if_string();
        if(problem == nullptr)
            MIGRAPHX_THROW("MLIR tuning problem must be a string");
        result.tuning_problem = *problem;
        result.tuning_solutions.resize(config.solutions.size());
        std::transform(config.solutions.begin(),
                       config.solutions.end(),
                       result.tuning_solutions.begin(),
                       [](const auto& solution) {
                           const auto* str = solution.if_string();
                           if(str == nullptr)
                               MIGRAPHX_THROW("MLIR tuning solution must be a string");
                           return *str;
                       });
        result.tuning_solution_views.reserve(result.tuning_solutions.size());
        std::transform(result.tuning_solutions.begin(),
                       result.tuning_solutions.end(),
                       std::back_inserter(result.tuning_solution_views),
                       &make_view);
        result.detailed_problem_info = config.detailed_problem_info;
    });
}

static mlir_backend_result* plugin_dump_mlir_to_mxr(const module* m,
                                                    const instruction_ref* inputs,
                                                    std::size_t input_count,
                                                    const fs::path::value_type* location,
                                                    std::size_t location_size) noexcept
{
    return make_backend_result([&](auto&) {
        dump_mlir_to_mxr(required(m, "module"),
                         copy_range(inputs, input_count),
                         copy_path(location, location_size));
    });
}

static mlir_backend_result* plugin_mlir_lds_usage_fits_arch(std::int64_t gemm_o,
                                                            const char* arch,
                                                            std::size_t arch_size,
                                                            shape::type_t elem_type,
                                                            const module* m) noexcept
{
    return make_backend_result([&](auto& result) {
        result.bool_value =
            mlir_lds_usage_fits_arch(gemm_o, copy_string(arch, arch_size), elem_type, m);
    });
}

static mlir_backend_string_view
plugin_result_error(const mlir_backend_result* result) noexcept
{
    assert(result != nullptr);
    return make_view(result->error);
}

static mlir_backend_string_view
plugin_result_string(const mlir_backend_result* result) noexcept
{
    assert(result != nullptr);
    return make_view(result->data);
}

static bool plugin_result_bool(const mlir_backend_result* result) noexcept
{
    assert(result != nullptr);
    return result->bool_value;
}

static mlir_backend_code_object_view
plugin_result_code_object(const mlir_backend_result* result) noexcept
{
    assert(result != nullptr);
    assert(result->code_object != nullptr);
    const auto& object = *result->code_object;
    return {object.cop.code_object.data(),
            object.cop.code_object.size(),
            make_view(object.cop.symbol_name),
            object.cop.global,
            object.cop.local,
            object.cop.expected_inputs.data(),
            object.cop.expected_inputs.size(),
            &object.cop.output,
            object.prefill_indices.data(),
            result->prefill_values.data(),
            object.prefill_indices.size()};
}

static mlir_backend_tuning_config_view
plugin_result_tuning_config(const mlir_backend_result* result) noexcept
{
    assert(result != nullptr);
    return {make_view(result->tuning_problem),
            result->tuning_solution_views.data(),
            result->tuning_solution_views.size(),
            make_view(result->detailed_problem_info)};
}

static void plugin_result_destroy(mlir_backend_result* result) noexcept { delete result; }

extern "C" MIGRAPHX_MLIR_PLUGIN_EXPORT const mlir_backend_v3*
migraphx_gpu_get_mlir_backend_v3() noexcept
{
    static const mlir_backend_v3 backend = {
        mlir_backend_abi_version,
        sizeof(mlir_backend_v3),
        &plugin_dump_mlir,
        &plugin_dump_mlir_to_file,
        &plugin_is_module_fusible,
        &plugin_compile_mlir,
        &plugin_get_tuning_config_mlir,
        &plugin_dump_mlir_to_mxr,
        &plugin_mlir_lds_usage_fits_arch,
        &plugin_result_error,
        &plugin_result_string,
        &plugin_result_bool,
        &plugin_result_code_object,
        &plugin_result_tuning_config,
        &plugin_result_destroy,
    };
    return &backend;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
