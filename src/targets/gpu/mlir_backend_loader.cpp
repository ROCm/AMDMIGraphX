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

// Keep MIGraphX's MLIR-facing API in migraphx_gpu while forwarding the
// rocMLIR-dependent work to the selected backend plugin.

#include <migraphx/gpu/mlir.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/env.hpp>
#include <migraphx/fileutils.hpp>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#ifdef MIGRAPHX_MLIR
#include <migraphx/gpu/mlir_backend.hpp>
#include <migraphx/dynamic_loader.hpp>
#include <migraphx/errors.hpp>
#include <iostream>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_MLIR_BACKEND);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_MLIR);

#ifdef MIGRAPHX_MLIR

static bool is_valid_mlir_backend(const mlir_backend_v3* table)
{
    return table != nullptr and table->abi_version == mlir_backend_abi_version and
           table->struct_size >= sizeof(mlir_backend_v3) and table->dump_mlir != nullptr and
           table->dump_mlir_to_file != nullptr and table->is_module_fusible != nullptr and
           table->compile_mlir != nullptr and table->get_tuning_config_mlir != nullptr and
           table->dump_mlir_to_mxr != nullptr and
           table->mlir_lds_usage_fits_arch != nullptr and table->result_error != nullptr and
           table->result_string != nullptr and table->result_bool != nullptr and
           table->result_code_object != nullptr and table->result_tuning_config != nullptr and
           table->result_destroy != nullptr;
}

// Load one backend for the process and retain the DLL for as long as its
// function table can be used.
static const mlir_backend_v3* load_mlir_backend()
{
    static dynamic_loader loader;
    static const mlir_backend_v3* vtable = []() -> const mlir_backend_v3* {
        auto backend = string_value_of(MIGRAPHX_MLIR_BACKEND{}, "legacy");
        if(backend.empty())
            backend = "legacy";
        const auto file = make_shared_object_filename("migraphx_mlir_" + backend);

        std::vector<fs::path> candidates;
        try
        {
            auto self_dir =
                dynamic_loader::path(reinterpret_cast<void*>(&load_mlir_backend)).parent_path();
            if(not self_dir.empty())
                candidates.push_back(self_dir / file);
        }
        catch(const std::exception&)
        {
        }
        candidates.emplace_back(file);

        for(const auto& candidate : candidates)
        {
            auto loaded = dynamic_loader::try_load(candidate);
            if(not loaded)
                continue;
            try
            {
                auto getter = loaded->get_function<const mlir_backend_v3*()>(
                    MIGRAPHX_GPU_MLIR_BACKEND_FACTORY_NAME);
                const auto* table = getter();
                if(not is_valid_mlir_backend(table))
                    MIGRAPHX_THROW("Invalid MLIR backend plugin interface");
                loader = *loaded;
                if(enabled(MIGRAPHX_TRACE_MLIR{}))
                    std::cout << "Loaded MLIR backend plugin: " << candidate.string() << std::endl;
                return table;
            }
            catch(const std::exception& e)
            {
                if(enabled(MIGRAPHX_TRACE_MLIR{}))
                    std::cout << "Failed to use MLIR backend plugin " << candidate.string() << ": "
                              << e.what() << std::endl;
            }
        }
        return nullptr;
    }();
    return vtable;
}

static const mlir_backend_v3& mlir_backend()
{
    const auto* table = load_mlir_backend();
    if(table == nullptr)
        MIGRAPHX_THROW("Failed to load MLIR backend plugin. Set MIGRAPHX_MLIR_BACKEND to "
                       "'legacy' or 'triton' and ensure the matching migraphx_mlir_*.dll is "
                       "next to migraphx_gpu.");
    return *table;
}

struct mlir_backend_result_deleter
{
    const mlir_backend_v3* backend = nullptr;

    void operator()(mlir_backend_result* result) const noexcept
    {
        if(result != nullptr)
            backend->result_destroy(result);
    }
};

using mlir_backend_result_ptr =
    std::unique_ptr<mlir_backend_result, mlir_backend_result_deleter>;

static mlir_backend_result_ptr checked_result(mlir_backend_result* result)
{
    const auto& backend = mlir_backend();
    mlir_backend_result_ptr handle{result, {&backend}};
    if(handle == nullptr)
        MIGRAPHX_THROW("MLIR backend plugin failed to return a result");

    const auto error = backend.result_error(handle.get());
    if(error.size != 0)
    {
        if(error.data == nullptr)
            MIGRAPHX_THROW("MLIR backend plugin returned an invalid error");
        MIGRAPHX_THROW(std::string{error.data, error.size});
    }
    return handle;
}

static std::string copy_string(mlir_backend_string_view view, const char* name)
{
    if(view.size == 0)
        return {};
    if(view.data == nullptr)
        MIGRAPHX_THROW(std::string{"MLIR backend plugin returned invalid "} + name);
    return {view.data, view.size};
}

template <class T>
static std::vector<T> copy_range(const T* data, std::size_t size, const char* name)
{
    if(size == 0)
        return {};
    if(data == nullptr)
        MIGRAPHX_THROW(std::string{"MLIR backend plugin returned invalid "} + name);
    return {data, data + size};
}

static value::binary
copy_binary(const std::uint8_t* data, std::size_t size, const char* name)
{
    if(size == 0)
        return {};
    if(data == nullptr)
        MIGRAPHX_THROW(std::string{"MLIR backend plugin returned invalid "} + name);
    return value::binary{data, size};
}

static std::string result_string(const mlir_backend_result_ptr& result)
{
    return copy_string(mlir_backend().result_string(result.get()), "string");
}

static const std::string& solution_string(const value& solution)
{
    const auto* result = solution.if_string();
    if(result == nullptr)
        MIGRAPHX_THROW("MLIR tuning solution must be a string");
    return *result;
}

static mlir_code_object result_code_object(const mlir_backend_result_ptr& result)
{
    const auto v = mlir_backend().result_code_object(result.get());
    if(v.output == nullptr)
        MIGRAPHX_THROW("MLIR backend plugin returned no output shape");

    mlir_code_object object;
    object.cop.code_object     = copy_binary(v.code_object, v.code_object_size, "code object");
    object.cop.symbol_name     = copy_string(v.symbol_name, "code object symbol");
    object.cop.global          = v.global;
    object.cop.local           = v.local;
    object.cop.expected_inputs =
        copy_range(v.expected_inputs, v.expected_input_count, "input shapes");
    object.cop.output      = *v.output;
    object.prefill_indices = copy_range(v.prefill_indices, v.prefill_count, "prefill indices");
    const auto prefill_values =
        copy_range(v.prefill_values, v.prefill_count, "prefill values");
    object.prefill_values.resize(prefill_values.size());
    std::transform(prefill_values.begin(),
                   prefill_values.end(),
                   object.prefill_values.begin(),
                   // Parentheses: value{T} prefers initializer_list and wraps a scalar as an array.
                   [](auto x) { return value(x); });
    return object;
}

static tuning_config result_tuning_config(const mlir_backend_result_ptr& result)
{
    const auto v = mlir_backend().result_tuning_config(result.get());
    tuning_config config;
    config.problem = copy_string(v.problem, "tuning problem");
    const auto solution_views =
        copy_range(v.solutions, v.solution_count, "tuning solutions");
    config.solutions.resize(solution_views.size());
    std::transform(solution_views.begin(),
                   solution_views.end(),
                   config.solutions.begin(),
                   [](auto solution) {
                       return value(copy_string(solution, "tuning solution"));
                   });
    config.detailed_problem_info =
        copy_string(v.detailed_problem_info, "detailed problem information");
    return config;
}

std::string dump_mlir(module m, const std::vector<shape>& inputs)
{
    const auto& backend = mlir_backend();
    auto result = checked_result(backend.dump_mlir(&m, inputs.data(), inputs.size()));
    return result_string(result);
}

std::string dump_mlir(module m)
{
    auto result = checked_result(mlir_backend().dump_mlir(&m, nullptr, 0));
    return result_string(result);
}

void dump_mlir_to_file(module m, const std::vector<shape>& inputs, const fs::path& location)
{
    const auto& path    = location.native();
    const auto& backend = mlir_backend();
    checked_result(
        backend.dump_mlir_to_file(&m, inputs.data(), inputs.size(), path.data(), path.size()));
}

bool is_module_fusible(const module& m, const context& migraphx_ctx, const value& solution)
{
    const auto& key     = solution_string(solution);
    const auto& backend = mlir_backend();
    auto result =
        checked_result(backend.is_module_fusible(&m, &migraphx_ctx, key.data(), key.size()));
    return backend.result_bool(result.get());
}

mlir_code_object compile_mlir(const context& migraphx_ctx,
                              module m,
                              const std::vector<shape>& in_shapes,
                              const value& solution)
{
    const auto& key     = solution_string(solution);
    const auto& backend = mlir_backend();
    auto result         = checked_result(backend.compile_mlir(
        &migraphx_ctx, &m, in_shapes.data(), in_shapes.size(), key.data(), key.size()));
    return result_code_object(result);
}

tuning_config get_tuning_config_mlir(const context& migraphx_ctx,
                                     module m,
                                     const std::vector<shape>& inputs,
                                     bool exhaustive)
{
    const auto& backend = mlir_backend();
    auto result         = checked_result(backend.get_tuning_config_mlir(
        &migraphx_ctx, &m, inputs.data(), inputs.size(), exhaustive));
    return result_tuning_config(result);
}

void dump_mlir_to_mxr(module m,
                      const std::vector<instruction_ref>& inputs,
                      const fs::path& location)
{
    const auto& path    = location.native();
    const auto& backend = mlir_backend();
    checked_result(
        backend.dump_mlir_to_mxr(&m, inputs.data(), inputs.size(), path.data(), path.size()));
}

bool mlir_lds_usage_fits_arch(int64_t gemm_o,
                              const std::string& arch,
                              shape::type_t elem_type,
                              const module* m)
{
    const auto& backend = mlir_backend();
    auto result         = checked_result(
        backend.mlir_lds_usage_fits_arch(gemm_o, arch.data(), arch.size(), elem_type, m));
    return backend.result_bool(result.get());
}

#else // MIGRAPHX_MLIR

template <class T>
void use(T&)
{
}

std::string dump_mlir(module) { return {}; }

std::string dump_mlir(module m, const std::vector<shape>& inputs)
{
    use(m);
    use(inputs);
    return {};
}

// NOLINTBEGIN(performance-unnecessary-value-param)
mlir_code_object compile_mlir(const context&, module, const std::vector<shape>&, const value&)
{
    return {};
}

tuning_config get_tuning_config_mlir(const context&, module, const std::vector<shape>&, bool)
{
    return {};
}

bool mlir_lds_usage_fits_arch(int64_t, const std::string&, shape::type_t, const module*)
{
    return false;
}

bool is_module_fusible(const module&, const context&, const value&) { return false; }

void dump_mlir_to_file(module, const std::vector<shape>&, const fs::path&) {}

void dump_mlir_to_mxr(module, const std::vector<instruction_ref>&, const fs::path&) {}
// NOLINTEND(performance-unnecessary-value-param)

#endif // MIGRAPHX_MLIR

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
