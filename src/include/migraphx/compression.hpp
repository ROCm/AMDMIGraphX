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
 
// zstd compression of whole buffers.
//
// Everything written is a single standard zstd frame (RFC 8878) recording the size of its
// content and a checksum of it, so the stock zstd tool can read it and damage is reported as an
// error rather than decoded into wrong bytes. zstd.h is only included by compression.cpp, so the
// dependency does not reach users of this header.

/// zstd's own default: about the ratio of zlib's default, at a fraction of the time.
constexpr int default_zstd_level = 3;

/// The largest content zstd_decompress produces unless told otherwise. A frame records its own
/// size, so this bounds what a damaged size field can make it allocate.
constexpr std::size_t default_zstd_max_size = std::size_t{256} << 20u;

/// Compress into one zstd frame.
MIGRAPHX_EXPORT std::vector<char>
zstd_compress(const char* data, std::size_t size, int level = default_zstd_level);

inline std::vector<char> zstd_compress(const std::vector<char>& data,
                                       int level = default_zstd_level)
{
    return zstd_compress(data.data(), data.size(), level);
}

/// Decompress exactly one zstd frame. Throws when the input is not one whole frame, is damaged
/// (including a checksum mismatch), needs a dictionary, or holds more than max_size bytes.
MIGRAPHX_EXPORT std::vector<char>
zstd_decompress(const char* data, std::size_t size, std::size_t max_size = default_zstd_max_size);

inline std::vector<char> zstd_decompress(const std::vector<char>& data,
                                         std::size_t max_size = default_zstd_max_size)
{
    return zstd_decompress(data.data(), data.size(), max_size);
}

/// True when the data starts with the magic number of a zstd frame.
MIGRAPHX_EXPORT bool is_zstd_frame(const char* data, std::size_t size);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_COMPRESSION_HPP
