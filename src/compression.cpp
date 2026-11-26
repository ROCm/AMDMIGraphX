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
#include <migraphx/compression.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/manage_ptr.hpp>
#include <zstd.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

using cctx_ptr = MIGRAPHX_MANAGE_PTR(ZSTD_CCtx*, ZSTD_freeCCtx);

/// The result of a zstd call, or an exception naming what failed.
std::size_t check(std::size_t r, const char* what)
{
    if(ZSTD_isError(r) != 0)
        MIGRAPHX_THROW(std::string{"zstd: "} + what + ": " + ZSTD_getErrorName(r));
    return r;
}

// The magic number that opens every zstd frame (RFC 8878 section 3.1.1), little endian.
constexpr std::array<char, 4> frame_magic = {'\x28', '\xb5', '\x2f', '\xfd'};

// Rejects skippable frames, which ZSTD_findFrameCompressedSize accepts and which would decode as
// empty.
bool is_zstd_frame(const char* data, std::size_t size)
{
    return size >= frame_magic.size() and std::equal(frame_magic.begin(), frame_magic.end(), data);
}

} // namespace

std::vector<char> zstd_compress(const char* data, std::size_t size)
{
    cctx_ptr cctx{ZSTD_createCCtx()};
    if(cctx == nullptr)
        MIGRAPHX_THROW("zstd: cannot create a compression context");
    // The checksum turns damage into an error on decompression instead of wrong bytes, and the
    // content size lets the reader allocate once and check it against its limit up front.
    check(ZSTD_CCtx_setParameter(cctx.get(), ZSTD_c_checksumFlag, 1), "checksum flag");
    check(ZSTD_CCtx_setParameter(cctx.get(), ZSTD_c_contentSizeFlag, 1), "content size flag");
    std::vector<char> out(ZSTD_compressBound(size));
    out.resize(check(ZSTD_compress2(cctx.get(), out.data(), out.size(), data, size), "compress"));
    return out;
}

std::vector<char> zstd_decompress(const char* data, std::size_t size, std::size_t max_size)
{
    if(not is_zstd_frame(data, size))
        MIGRAPHX_THROW("zstd: not a zstd frame");
    // Exactly one frame: zstd would otherwise also decode frames concatenated after the first,
    // and anything trailing it is a sign of damage.
    if(check(ZSTD_findFrameCompressedSize(data, size), "frame size") != size)
        MIGRAPHX_THROW("zstd: unexpected data after the frame");
    // The header was validated above, so the only failure left is a size that was not recorded.
    auto content = ZSTD_getFrameContentSize(data, size);
    if(content == ZSTD_CONTENTSIZE_UNKNOWN)
        MIGRAPHX_THROW("zstd: the frame does not record its content size");
    if(content > max_size)
        MIGRAPHX_THROW("zstd: content of " + std::to_string(content) + " bytes exceeds the " +
                       std::to_string(max_size) + " byte limit");

    // zstd itself rejects a frame that needs a dictionary or decodes to other than its recorded
    // size.
    std::vector<char> out(content);
    auto n = check(ZSTD_decompress(out.data(), out.size(), data, size), "decompress");
    assert(n == out.size());
    (void)n;
    return out;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
