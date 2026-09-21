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
#include <utility>

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

// Load one backend for the process and retain the DLL for as long as its
// function table can be used.
static const mlir_backend_v2* load_mlir_backend()
{
    static dynamic_loader loader;
    static const mlir_backend_v2* vtable = []() -> const mlir_backend_v2* {
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
                auto getter = loaded->get_function<const mlir_backend_v2*()>(
                    MIGRAPHX_GPU_MLIR_BACKEND_FACTORY_NAME);
                const auto* table = getter();
                if(table != nullptr)
                {
                    loader = *loaded;
                    if(enabled(MIGRAPHX_TRACE_MLIR{}))
                        std::cout << "Loaded MLIR backend plugin: " << candidate.string()
                                  << std::endl;
                    return table;
                }
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

static const mlir_backend_v2& mlir_backend()
{
    const auto* table = load_mlir_backend();
    if(table == nullptr)
        MIGRAPHX_THROW("Failed to load MLIR backend plugin. Set MIGRAPHX_MLIR_BACKEND to "
                       "'legacy' or 'triton' and ensure the matching migraphx_mlir_*.dll is "
                       "next to migraphx_gpu.");
    return *table;
}

std::string dump_mlir(module m, const std::vector<shape>& inputs)
{
    return mlir_backend().dump_mlir(std::move(m), inputs);
}

std::string dump_mlir(module m) { return mlir_backend().dump_mlir(std::move(m), {}); }

void dump_mlir_to_file(module m, const std::vector<shape>& inputs, const fs::path& location)
{
    mlir_backend().dump_mlir_to_file(std::move(m), inputs, location);
}

bool is_module_fusible(const module& m, const context& migraphx_ctx, const value& solution)
{
    return mlir_backend().is_module_fusible(m, migraphx_ctx, solution);
}

mlir_code_object compile_mlir(const context& migraphx_ctx,
                              module m,
                              const std::vector<shape>& in_shapes,
                              const value& solution)
{
    return mlir_backend().compile_mlir(migraphx_ctx, std::move(m), in_shapes, solution);
}

tuning_config get_tuning_config_mlir(const context& migraphx_ctx,
                                     module m,
                                     const std::vector<shape>& inputs,
                                     bool exhaustive)
{
    return mlir_backend().get_tuning_config_mlir(migraphx_ctx, std::move(m), inputs, exhaustive);
}

void dump_mlir_to_mxr(module m,
                      const std::vector<instruction_ref>& inputs,
                      const fs::path& location)
{
    mlir_backend().dump_mlir_to_mxr(std::move(m), inputs, location);
}

bool mlir_lds_usage_fits_arch(int64_t gemm_o,
                              const std::string& arch,
                              shape::type_t elem_type,
                              const module* m)
{
    return mlir_backend().mlir_lds_usage_fits_arch(gemm_o, arch, elem_type, m);
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
