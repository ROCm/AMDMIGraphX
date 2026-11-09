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
#ifndef MIGRAPHX_GUARD_KERNELS_BUFFER_LOAD_HPP
#define MIGRAPHX_GUARD_KERNELS_BUFFER_LOAD_HPP

#include <migraphx/kernels/bit_cast.hpp>
#include <migraphx/kernels/vec.hpp>
#include <migraphx/kernels/types.hpp>

namespace migraphx {

// gfx12 buffer-resource word 3 (from composable_kernel): makes raw buffer loads
// return 0 for out-of-range byte offsets, so bounds/halo checks collapse to an
// offset select against a sentinel instead of a per-load branch.
constexpr uint32_t oob_buffer_rsrc_word3 = 0x31004000;

// Build an out-of-bounds-tolerant buffer descriptor for a read-only pointer (the
// resource is only ever loaded from; the const_cast is required by the builtin's
// non-const pointer parameter).
template <class T>
__device__ inline __amdgpu_buffer_rsrc_t make_oob_buffer_rsrc(const T* p, uint32_t byte_count)
{
    auto* base = const_cast<T*>(p); // NOLINT(cppcoreguidelines-pro-type-const-cast)
    return __builtin_amdgcn_make_buffer_rsrc(base, 0, byte_count, oob_buffer_rsrc_word3);
}

// Raw buffer load of N contiguous T (N*sizeof(T) must be 2/4/8/16 bytes ->
// b16/b32/b64/b128; gfx12 tolerates 4-byte alignment). OOB bytes read as 0.
template <class T, index_int N>
__device__ inline vec<T, N> buffer_load_vec(__amdgpu_buffer_rsrc_t rsrc, int byte_offset)
{
    constexpr index_int bytes = N * sizeof(T);
    static_assert(bytes == 2 or bytes == 4 or bytes == 8 or bytes == 16,
                  "buffer_load_vec width must be 2, 4, 8, or 16 bytes");
    if constexpr(bytes == 16)
        return bit_cast<vec<T, N>>(__builtin_amdgcn_raw_buffer_load_b128(rsrc, byte_offset, 0, 0));
    else if constexpr(bytes == 8)
        return bit_cast<vec<T, N>>(__builtin_amdgcn_raw_buffer_load_b64(rsrc, byte_offset, 0, 0));
    else if constexpr(bytes == 4)
        return bit_cast<vec<T, N>>(__builtin_amdgcn_raw_buffer_load_b32(rsrc, byte_offset, 0, 0));
    else
        return bit_cast<vec<T, N>>(__builtin_amdgcn_raw_buffer_load_b16(rsrc, byte_offset, 0, 0));
}

// Raw buffer load of a single T (2- or 4-byte element). OOB reads as 0.
template <class T>
__device__ inline T buffer_load(__amdgpu_buffer_rsrc_t rsrc, int byte_offset)
{
    static_assert(sizeof(T) == 2 or sizeof(T) == 4, "buffer_load element must be 2 or 4 bytes");
    if constexpr(sizeof(T) == 2)
        return bit_cast<T>(__builtin_amdgcn_raw_buffer_load_b16(rsrc, byte_offset, 0, 0));
    else
        return bit_cast<T>(__builtin_amdgcn_raw_buffer_load_b32(rsrc, byte_offset, 0, 0));
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_BUFFER_LOAD_HPP
