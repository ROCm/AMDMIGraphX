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
 *
 */
#ifndef MIGRAPHX_GUARD_GPU_BINARY_CACHE_STATS_HPP
#define MIGRAPHX_GUARD_GPU_BINARY_CACHE_STATS_HPP

#include <migraphx/gpu/config.hpp>
#include <migraphx/value.hpp>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

/// The model a program compile belongs to, as recorded in the statistics tables. Two compiles
/// are the same model when both the name and the program hash match.
struct binary_cache_model
{
    std::string name         = {};
    /// Digest of the program as it was handed to compile, without weight data.
    std::string program_hash = {};
    /// Parameter names and shapes, as a JSON object.
    value inputs = value::object{};
    /// The compile options and backend options, as a JSON object.
    value options = value::object{};
};

/// How a kernel was used since the last time the statistics were written. Signed, since the
/// counts are bound to SQLite integers.
struct binary_cache_usage
{
    /// Instructions that were given this kernel, including the one it was first compiled for.
    std::int64_t uses = 0;
    /// Times it was loaded from storage instead of being compiled.
    std::int64_t hits = 0;
};

/// Usage by compile key.
using binary_cache_usage_map = std::unordered_map<std::string, binary_cache_usage>;

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_BINARY_CACHE_STATS_HPP
