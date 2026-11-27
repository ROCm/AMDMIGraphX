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
#include <migraphx/gpu/mlir_perf_config.hpp>
#include <migraphx/stringutils.hpp>
#include <algorithm>
#include <iterator>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

namespace {

// rocMLIR's schema defaults for the fields this file reads; anything else has no default here.
std::optional<long> schema_default(const std::string& key)
{
    if(key == "numCTAs")
        return 1;
    if(key == "numWaves")
        return 4;
    if(key == "useWarpSpecialize")
        return -1;
    return std::nullopt;
}

// The comma-separated "key=value" tokens after the first ':', or nothing if there is no ':'.
std::optional<std::vector<std::string>> perf_config_tokens(const std::string& config)
{
    auto pos = config.find(':');
    if(pos == std::string::npos)
        return std::nullopt;
    auto rest = config.substr(pos + 1);
    if(rest.empty())
        return std::nullopt;
    return split_string(rest, ',');
}

} // namespace

bool is_named_perf_config(const std::string& config)
{
    auto tokens = perf_config_tokens(config);
    if(not tokens.has_value())
        return false;
    return std::all_of(tokens->begin(), tokens->end(), [](const std::string& t) {
        return t.find('=') != std::string::npos;
    });
}

std::string perf_config_prefix(const std::string& config)
{
    auto pos = config.find(':');
    return pos == std::string::npos ? std::string{} : config.substr(0, pos);
}

std::optional<long> get_perf_config_field(const std::string& config, const std::string& key)
{
    if(not is_named_perf_config(config))
        return std::nullopt;
    auto tokens = perf_config_tokens(config);
    for(const auto& token : *tokens)
    {
        auto eq = token.find('=');
        if(eq == std::string::npos or token.substr(0, eq) != key)
            continue;
        try
        {
            return std::stol(token.substr(eq + 1));
        }
        catch(...)
        {
            return std::nullopt;
        }
    }
    return schema_default(key);
}

std::string set_perf_config_field(const std::string& config, const std::string& key, long value)
{
    auto tokens = perf_config_tokens(config);
    if(not tokens.has_value() or
       not std::all_of(tokens->begin(), tokens->end(), [](const std::string& t) {
           return t.find('=') != std::string::npos;
       }))
        return config;
    auto prefix  = perf_config_prefix(config);
    bool found   = false;
    auto new_val = key + "=" + std::to_string(value);
    for(auto& token : *tokens)
    {
        auto eq = token.find('=');
        if(eq != std::string::npos and token.substr(0, eq) == key)
        {
            token = new_val;
            found = true;
            break;
        }
    }
    if(not found)
        tokens->push_back(new_val);
    return prefix + ":" + join_strings(*tokens, ",");
}

bool is_tile_splittable(long m_per_block, long n_per_block, long num_ctas)
{
    if(num_ctas <= 1)
        return true;
    long split_m        = 1;
    long split_n        = 1;
    bool split_along_m  = true;
    const long min_wmma = 16;
    while(split_m * split_n < num_ctas)
    {
        (split_along_m ? split_m : split_n) *= 2;
        split_along_m = not split_along_m;
    }
    return (m_per_block / split_m >= min_wmma) and (n_per_block / split_n >= min_wmma);
}

std::vector<std::string>
apply_num_ctas(const std::vector<std::string>& solutions, long request, long max_cluster)
{
    if(request == 0)
        return solutions;

    if(request == 1)
    {
        std::vector<std::string> result;
        std::copy_if(solutions.begin(),
                     solutions.end(),
                     std::back_inserter(result),
                     [](const std::string& s) {
                         return get_perf_config_field(s, "numCTAs").value_or(1) == 1;
                     });
        return result;
    }

    // A tile-splittable candidate's clustered copy is offered first, the unclustered original
    // right after as a fallback: nothing here asks whether the device can actually honor
    // clustering (no device check happens in this file), so rocMLIR may reject the clustered
    // candidate. compile_ops.cpp's single-candidate (eager/skip-benchmark) path now retries the
    // rest of the list in order when the first candidate fails to compile, so the preferred
    // clustered candidate can be tried first while still falling back to the always-compilable
    // original; exhaustive/benchmarked compiles still race both fairly and drop the clustered one
    // gracefully if rocMLIR rejects it.
    auto clamped = std::min(request, max_cluster);
    std::vector<std::string> result;
    std::unordered_set<std::string> seen;
    auto add = [&](const std::string& s) {
        if(seen.insert(s).second)
            result.push_back(s);
    };
    for(const auto& s : solutions)
    {
        if(is_named_perf_config(s) and perf_config_prefix(s) == "gemm" and
           get_perf_config_field(s, "numCTAs").value_or(1) == 1)
        {
            auto m = get_perf_config_field(s, "mPerBlock");
            auto n = get_perf_config_field(s, "nPerBlock");
            if(m.has_value() and n.has_value() and is_tile_splittable(*m, *n, clamped))
                add(set_perf_config_field(s, "numCTAs", clamped));
        }
        add(s);
    }
    return result;
}

bool is_warp_specialize_eligible(const std::string& solution)
{
    if(not is_named_perf_config(solution) or perf_config_prefix(solution) != "gemm")
        return false;
    auto waves = get_perf_config_field(solution, "numWaves").value_or(4);
    auto ctas  = get_perf_config_field(solution, "numCTAs").value_or(1);
    auto ws    = get_perf_config_field(solution, "useWarpSpecialize").value_or(-1);
    return waves == 4 and ctas == 1 and ws != 1;
}

std::vector<std::string> apply_warp_specialize(const std::vector<std::string>& solutions,
                                               warp_specialize_mode mode)
{
    if(mode == warp_specialize_mode::off)
        return solutions;

    if(mode == warp_specialize_mode::force)
    {
        // Each eligible candidate's warp-specialized copy is offered first, the unmodified
        // original right after as a fallback: rewriting eligible candidates in place with no
        // fallback would leave compile_ops.cpp's single-candidate (eager/skip-benchmark) path
        // with nothing to retry if rocMLIR rejects warp specialization on this device.
        std::vector<std::string> result;
        result.reserve(solutions.size() * 2);
        for(const auto& s : solutions)
        {
            if(is_warp_specialize_eligible(s))
                result.push_back(set_perf_config_field(s, "useWarpSpecialize", 1));
            result.push_back(s);
        }
        return result;
    }

    // tune: originals first, then one useWarpSpecialize=1 copy of each eligible candidate.
    std::vector<std::string> result = solutions;
    for(const auto& s : solutions)
    {
        if(is_warp_specialize_eligible(s))
            result.push_back(set_perf_config_field(s, "useWarpSpecialize", 1));
    }
    return result;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
