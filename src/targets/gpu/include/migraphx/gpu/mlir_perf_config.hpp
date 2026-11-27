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
#ifndef MIGRAPHX_GUARD_GPU_MLIR_PERF_CONFIG_HPP
#define MIGRAPHX_GUARD_GPU_MLIR_PERF_CONFIG_HPP

#include <migraphx/gpu/config.hpp>
#include <migraphx/enum.hpp>
#include <optional>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// Pure string helpers for rocMLIR's named perf-config strings, e.g.
//   "gemm:mPerBlock=128,nPerBlock=128,kPerBlock=64,kpack=1,numCTAs=1,numWaves=4,...,
//    useWarpSpecialize=-1"
// No MLIR dependency, so these are unit-testable without a GPU.

// True when `config` is a named perf-config string ("prefix:key=value,key=value,...") rather
// than a legacy positional one ("prefix:v1:..."). Only named configs are understood below; a
// positional (or otherwise unrecognized) string is left unchanged by every transform here.
MIGRAPHX_GPU_EXPORT bool is_named_perf_config(const std::string& config);

// The prefix of a perf-config string, e.g. "gemm" from "gemm:mPerBlock=...", or "attn" from an
// attention config. Empty when `config` has no ':'.
MIGRAPHX_GPU_EXPORT std::string perf_config_prefix(const std::string& config);

// The integer value of `key` in a named perf-config string. When the field is absent, returns
// rocMLIR's schema default for the few keys this file cares about (numCTAs=1, numWaves=4,
// useWarpSpecialize=-1); nothing for any other absent key or a non-named config.
MIGRAPHX_GPU_EXPORT std::optional<long>
get_perf_config_field(const std::string& config, const std::string& key);

// `config` with `key` set to `value`: replaces the field if present, appends it otherwise.
// Returns `config` unchanged when it is not a named perf config.
MIGRAPHX_GPU_EXPORT std::string
set_perf_config_field(const std::string& config, const std::string& key, long value);

// rocMLIR's isTileSplittableAcrossCTAs: true without splitting anything when num_ctas <= 1;
// otherwise doubles the M split, then the N split, alternately, until the product reaches
// num_ctas, and requires both resulting per-CTA tile dimensions to be at least 16.
MIGRAPHX_GPU_EXPORT bool is_tile_splittable(long m_per_block, long n_per_block, long num_ctas);

// Rewrites a candidate list (rocMLIR's quick or exhaustive tuning solutions) for a
// `mlir_num_ctas` backend option:
//   0: left unchanged (rocMLIR's own list; clusters only ever appear in exhaustive tuning)
//   1: keep only the candidates whose numCTAs is already 1 (the default: no clusters)
//   N (2, 4, 8 or 16): each named "gemm:" candidate at numCTAs=1 whose tile is splittable by N
//      gets a numCTAs=N copy offered first (preferred when only the first candidate is used,
//      e.g. an eager compile), with the unclustered original kept right after as a fallback.
//      Every other original candidate is kept unchanged too. Nothing here checks whether the
//      device can actually honor clustering, so rocMLIR may reject every clustered candidate;
//      keeping the originals means that still leaves a compilable kernel instead of zero
//      candidates. Duplicate strings are removed, preserving first appearance.
MIGRAPHX_GPU_EXPORT std::vector<std::string>
apply_num_ctas(const std::vector<std::string>& solutions, long request, long max_cluster);

// off: never request warp specialization. tune: append a useWarpSpecialize=1 copy of each
// eligible candidate, so the benchmark decides. force: rewrite every eligible candidate in place.
//
// Declared as a nested enum (as layout_convolution::layout_order is) rather than a namespace-
// scope MIGRAPHX_ENUM_CLASS: the generated to_string/from_string/operator<< are then friend
// functions found only via ADL, so they do not shadow migraphx::to_string's general overloads at
// every other unqualified `to_string(...)` call site in this namespace.
struct MIGRAPHX_GPU_EXPORT warp_specialize_mode_holder
{
    MIGRAPHX_NESTED_ENUM(warp_specialize_mode, off, tune, force)
};
using warp_specialize_mode = warp_specialize_mode_holder::warp_specialize_mode;

// A candidate is eligible for warp specialization when it is a named "gemm:" config with
// numWaves=4, numCTAs=1, and useWarpSpecialize not already 1. (Whether the module itself has an
// eligible fp16 dot/convolution is a separate, MLIR-aware check left to the caller.)
MIGRAPHX_GPU_EXPORT bool is_warp_specialize_eligible(const std::string& solution);

// Rewrites a candidate list for a `mlir_warp_specialize` backend option; see warp_specialize_mode.
MIGRAPHX_GPU_EXPORT std::vector<std::string>
apply_warp_specialize(const std::vector<std::string>& solutions, warp_specialize_mode mode);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_MLIR_PERF_CONFIG_HPP
