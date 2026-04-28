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
#include <migraphx/gpu/amdgpu_kernel_registry.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/env.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/gpu/context.hpp>

#include <nlohmann/json.hpp>
#include <fstream>
#include <algorithm>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_AMDGPU_KERNEL_REGISTRY)

namespace {
using json = nlohmann::json;

struct kernel_entry
{
    std::string op_name;
    std::string dtype;
    std::vector<std::string> archs;
    std::string binary_path;
    std::string symbol_name;
    std::size_t local_size = 256;
};

struct parsed_registry
{
    std::string root_dir;
    std::vector<kernel_entry> entries;
};

static std::string normalize_op_name(std::string op_name)
{
    return replace_string(op_name, "::", ".");
}

static std::string shape_type_to_registry_dtype(shape::type_t t)
{
    if(t == shape::half_type)
        return "float16";
    if(t == shape::bf16_type)
        return "bfloat16";
    if(t == shape::float_type)
        return "float32";
    return "";
}

static std::string resolve_registry_path(const operation& op)
{
    const auto opv = op.to_value();
    const auto p   = opv.get("kernel_registry", std::string{});
    if(not p.empty())
        return p;
    return string_value_of(MIGRAPHX_AMDGPU_KERNEL_REGISTRY{});
}

static parsed_registry parse_registry_file(const std::string& path)
{
    std::ifstream fs(path);
    if(not fs)
        MIGRAPHX_THROW("AMDGPU registry: unable to open registry file: " + path);

    json j;
    fs >> j;

    parsed_registry out;
    out.root_dir = fs::path(path).parent_path().string();

    if(not j.contains("kernels") or not j["kernels"].is_array())
        return out;

    for(const auto& k : j["kernels"])
    {
        const auto op_name = k.value("operation", "");
        if(op_name.empty())
            continue;
        if(not k.contains("binaries") or not k["binaries"].is_array())
            continue;

        for(const auto& b : k["binaries"])
        {
            kernel_entry e;
            e.op_name     = op_name;
            e.dtype       = b.value("dtype", "");
            e.binary_path = b.value("path", "");

            if(e.binary_path.empty())
                continue;

            if(b.contains("architectures") and b["architectures"].is_array())
            {
                for(const auto& a : b["architectures"])
                    e.archs.push_back(a.get<std::string>());
            }

            if(b.contains("metadata") and b["metadata"].is_object())
            {
                const auto& m = b["metadata"];
                if(m.contains("threads_per_block") and m["threads_per_block"].is_array() and
                   not m["threads_per_block"].empty())
                {
                    e.local_size = m["threads_per_block"][0].get<std::size_t>();
                }
                e.symbol_name = m.value("symbol_name", "");
            }

            if(e.symbol_name.empty())
                e.symbol_name = b.value("symbol_name", "");

            if(e.symbol_name.empty())
            {
                auto base = normalize_op_name(e.op_name);
                base = replace_string(base, ".", "_");
                e.symbol_name = base + "_kernel";
            }

            out.entries.push_back(std::move(e));
        }
    }

    return out;
}

static const parsed_registry& get_registry(const std::string& path)
{
    static std::unordered_map<std::string, parsed_registry> cache;
    auto it = cache.find(path);
    if(it == cache.end())
        it = cache.emplace(path, parse_registry_file(path)).first;
    return it->second;
}

static bool arch_matches(const kernel_entry& e, const std::string& gfx_name)
{
    if(e.archs.empty())
        return true;
    return std::find(e.archs.begin(), e.archs.end(), gfx_name) != e.archs.end();
}

} // namespace

amdgpu_kernel_match find_amdgpu_kernel(const context& ctx,
                                       const operation& op,
                                       const std::vector<shape>& inputs)
{
    amdgpu_kernel_match result;

    const auto registry_path = resolve_registry_path(op);
    if(registry_path.empty())
        return result;

    const auto dtype = inputs.empty() ? std::string{} : shape_type_to_registry_dtype(inputs[0].type());
    if(dtype.empty())
        return result;

    const auto op_name  = normalize_op_name(op.name());
    const auto gfx_name = ctx.get_current_device().get_gfx_name();

    const auto& registry = get_registry(registry_path);
    for(const auto& e : registry.entries)
    {
        if(e.op_name != op_name)
            continue;
        if(not e.dtype.empty() and e.dtype != dtype)
            continue;
        if(not arch_matches(e, gfx_name))
            continue;

        fs::path p{e.binary_path};
        if(p.is_relative())
            p = fs::path(registry.root_dir) / p;

        result.found       = true;
        result.binary_path = p.string();
        result.symbol_name = e.symbol_name;
        result.local_size  = e.local_size;
        return result;
    }

    return result;
}

bool has_amdgpu_kernel(const context& ctx, const operation& op, const std::vector<shape>& inputs)
{
    return find_amdgpu_kernel(ctx, op, inputs).found;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
