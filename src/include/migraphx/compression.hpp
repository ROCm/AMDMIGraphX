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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_COMPRESSION_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_COMPRESSION_HPP

#include <migraphx/config.hpp>
#include <cstddef>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// Each buffer is one standard zstd frame (RFC 8878) with its content size and a checksum, so the
// zstd tool can read it and damage is an error rather than wrong bytes. zstd.h stays private to
// compression.cpp.

/// The largest content zstd_decompress produces unless told otherwise. A frame records its own
/// size, so this bounds what a damaged size field can make it allocate.
constexpr std::size_t default_zstd_max_size = std::size_t{256} << 20u;

/// Compress into one zstd frame, at zstd's default level.
MIGRAPHX_EXPORT std::vector<char> zstd_compress(const char* data, std::size_t size);

inline std::vector<char> zstd_compress(const std::vector<char>& data)
{
    return zstd_compress(data.data(), data.size());
}

/// Decompress exactly one zstd frame that records its content size, at most max_size bytes.
/// Throws otherwise, or when the frame is damaged or needs a dictionary.
MIGRAPHX_EXPORT std::vector<char>
zstd_decompress(const char* data, std::size_t size, std::size_t max_size = default_zstd_max_size);

inline std::vector<char> zstd_decompress(const std::vector<char>& data,
                                         std::size_t max_size = default_zstd_max_size)
{
    return zstd_decompress(data.data(), data.size(), max_size);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_COMPRESSION_HPP
