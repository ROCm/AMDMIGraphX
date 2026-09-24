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
#ifndef MIGRAPHX_GUARD_RTGLIB_MD5_HPP
#define MIGRAPHX_GUARD_RTGLIB_MD5_HPP

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/// Incremental MD5. Feed bytes with update() in any number of pieces; the digest depends
/// only on the concatenation, so splitting the input never changes the result.
struct MIGRAPHX_EXPORT md5_hasher
{
    md5_hasher& update(const std::string_view& str);
    /// Digest of everything fed so far, as a lowercase hex string. The hasher is left
    /// untouched, so more bytes can still be added afterwards.
    std::string finalize() const;

    private:
    std::array<std::uint32_t, 4> state = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};
    std::array<std::uint8_t, 64> buffer{};
    std::size_t buffered     = 0;
    std::uint64_t total_size = 0;
};

/// Compute the MD5 digest of a string and return it as a lowercase hex string.
std::string MIGRAPHX_EXPORT md5(const std::string_view& str);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
