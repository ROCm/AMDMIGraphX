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
#ifndef MIGRAPHX_GUARD_GPU_FILE_BINARY_CACHE_HPP
#define MIGRAPHX_GUARD_GPU_FILE_BINARY_CACHE_HPP

#include <migraphx/gpu/config.hpp>
#include <migraphx/gpu/binary_cache_entry.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/optional.hpp>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// A binary_cache_backend that keeps entries as files under a root directory, laid out
// <root>/<version>/<device>/<key_hash>.mxr, with a cache.info stamp beside each version
// directory describing the build that produced it. Path resolution and the text of the stamp
// are the caller's job.
struct MIGRAPHX_GPU_EXPORT file_binary_cache
{
    optional<std::vector<char>>
    load(const std::string& version, const std::string& device, const std::string& key_hash);
    void store(const std::string& version,
               const std::string& device,
               const std::string& key_hash,
               const binary_cache_entry& e,
               const std::vector<char>& blob);

    fs::path root     = {};
    std::string stamp = {};
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_FILE_BINARY_CACHE_HPP
