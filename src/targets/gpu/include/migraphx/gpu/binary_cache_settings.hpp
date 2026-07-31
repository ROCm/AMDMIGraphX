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
#ifndef MIGRAPHX_GUARD_GPU_BINARY_CACHE_SETTINGS_HPP
#define MIGRAPHX_GUARD_GPU_BINARY_CACHE_SETTINGS_HPP

// What a caller needs to hold or configure a binary cache, separated from the cache itself so
// that context.hpp and compile_ops.hpp do not pull the IR headers into every GPU source file.

#include <migraphx/gpu/config.hpp>
#include <memory>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct binary_cache;

/// How the cache behaves for one compile. Supplied through the target's backend options, with
/// the MIGRAPHX_BINARY_CACHE environment variable as the default for the directory.
struct binary_cache_settings
{
    /// Where entries are kept. Nothing is written to disk when this is empty, though results
    /// are still shared in memory.
    std::string path = {};
    /// Compile even when a result can be reused, and fail if the two disagree.
    bool verify = false;

    MIGRAPHX_GPU_EXPORT static binary_cache_settings defaults();
};

/// Lets a context own a cache without seeing its definition.
MIGRAPHX_GPU_EXPORT std::shared_ptr<binary_cache> make_binary_cache();

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_BINARY_CACHE_SETTINGS_HPP
