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

// Host-side backend loader for the MLIR compilation backend. This file lives in
// migraphx_gpu and implements the public functions declared in mlir.hpp by
// forwarding the rocMLIR-touching ones to a dynamically loaded backend plugin
// (legacy rocMLIR or rocmlirTriton), selected at runtime via
// MIGRAPHX_MLIR_BACKEND. The pure-migraphx helpers (adjust_param_shapes,
// insert_mlir) are implemented directly here since they never touch rocMLIR.

#include <migraphx/gpu/mlir.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/env.hpp>
#include <algorithm>

#ifdef MIGRAPHX_MLIR
#include <migraphx/gpu/mlir_backend.hpp>
#include <migraphx/dynamic_loader.hpp>
#include <migraphx/errors.hpp>
#include <iostream>
#include <mutex>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_MLIR_BACKEND);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_MLIR);

// These two helpers never touch rocMLIR, so they are kept in migraphx_gpu and
// do not require the backend plugin.
void adjust_param_shapes(module& m, const std::vector<shape>& inputs)
{
    auto names = m.get_parameter_names();
    std::sort(names.begin(), names.end());
    for(auto i : range(names.size()))
    {
        const auto& name  = names[i];
        const auto& input = inputs[i];
        auto param        = m.get_parameter(name);
        assert(param->get_shape().standard());
        if(input.standard())
            continue;
        auto new_param = m.add_parameter(name + ".0", input);
        m.replace_instruction(param, new_param);
        m.remove_instruction(param);
    }
}

instruction_ref insert_mlir(module& m,
                            instruction_ref ins,
                            code_object_op co,
                            const std::vector<instruction_ref>& inputs)
{
    std::vector<instruction_ref> refs;
    std::size_t last = 0;
    refs.reserve(inputs.size());
    std::copy(inputs.begin(), inputs.end(), std::back_inserter(refs));
    last               = refs.size() - 1;
    co.expected_inputs = to_shapes(refs);
    co.output_arg      = last;
    return m.insert_instruction(ins, co, refs);
}

#ifdef MIGRAPHX_MLIR

static std::string plugin_file_name(const std::string& backend)
{
    const std::string base = "migraphx_mlir_" + backend;
#ifdef _WIN32
    return base + ".dll";
#else
    return "lib" + base + ".so";
#endif
}

// Loads the backend plugin selected by MIGRAPHX_MLIR_BACKEND exactly once and
// caches the vtable. The dynamic_loader is kept alive for the program lifetime
// so the returned pointer stays valid.
static const mlir_backend_v1* load_mlir_backend()
{
    static dynamic_loader loader;
    static const mlir_backend_v1* vtable = [&]() -> const mlir_backend_v1* {
        auto backend = string_value_of(MIGRAPHX_MLIR_BACKEND{}, "legacy");
        if(backend.empty())
            backend = "legacy";
        const auto file = plugin_file_name(backend);

        std::vector<fs::path> candidates;
        try
        {
            // Look next to the migraphx_gpu module first.
            auto self_dir =
                dynamic_loader::path(reinterpret_cast<void*>(&load_mlir_backend)).parent_path();
            if(not self_dir.empty())
                candidates.push_back(self_dir / file);
        }
        catch(const std::exception&)
        {
        }
        // Fall back to the default OS search path.
        candidates.emplace_back(file);

        for(const auto& candidate : candidates)
        {
            auto loaded = dynamic_loader::try_load(candidate);
            if(not loaded)
                continue;
            try
            {
                auto getter = loaded->get_function<const mlir_backend_v1*()>(
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

static const mlir_backend_v1& mlir_backend()
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

void dump_mlir_to_file(module, const std::vector<shape>&, const fs::path&) {}

bool is_module_fusible(const module&, const context&, const value&) { return false; }

// NOLINTBEGIN(performance-unnecessary-value-param)
mlir_code_object compile_mlir(const context&, module, const std::vector<shape>&, const value&)
{
    return {};
}

tuning_config get_tuning_config_mlir(const context&, module, const std::vector<shape>&, bool)
{
    return {};
}
// NOLINTEND(performance-unnecessary-value-param)

void dump_mlir_to_mxr(module, const std::vector<instruction_ref>&, const fs::path&) {}

#endif // MIGRAPHX_MLIR

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
