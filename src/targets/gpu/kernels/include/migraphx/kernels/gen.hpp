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
#ifndef MIGRAPHX_GUARD_KERNELS_GEN_HPP
#define MIGRAPHX_GUARD_KERNELS_GEN_HPP

#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/debug.hpp>

namespace migraphx {
namespace gen {

// ============================================================
// Vectorized memory access
// ============================================================

template <index_int N, class T>
__device__ auto vec_load(T* data, index_int offset)
{
    array<T, N> result;
    auto base = offset * N;
    for(index_int i = 0; i < N; ++i)
        result[i] = data[base + i];
    return result;
}

template <index_int N, class T, class V>
__device__ void vec_store(T* data, index_int offset, V value)
{
    auto base = offset * N;
    for(index_int i = 0; i < N; ++i)
        data[base + i] = value[i];
}

// ============================================================
// Strided memory access
// ============================================================

template <index_int N, index_int Stride, class T>
__device__ auto strided_load(T* data, index_int base)
{
    array<T, N> result;
    for(index_int i = 0; i < N; ++i)
        result[i] = data[base + i * Stride];
    return result;
}

template <index_int N, index_int Stride, class T, class V>
__device__ void strided_store(T* data, index_int base, V value)
{
    for(index_int i = 0; i < N; ++i)
        data[base + i * Stride] = value[i];
}

// ============================================================
// Conditional load with bounds check
// ============================================================

template <class T, class Fill>
__device__ auto conditional_load(T* data, int64_t offset, index_int size, Fill fill)
{
    if(offset >= 0 and static_cast<index_int>(offset) < size)
        return data[offset];
    return fill;
}

// ============================================================
// Per-lane reduction (array reduce)
// ============================================================

template <class Array>
__device__ auto lane_reduce_sum(Array a)
{
    auto result = a[0];
    for(index_int i = 1; i < a.size(); ++i)
        result = result + a[i];
    return result;
}

template <class Array>
__device__ auto lane_reduce_product(Array a)
{
    auto result = a[0];
    for(index_int i = 1; i < a.size(); ++i)
        result = result * a[i];
    return result;
}

template <class Array>
__device__ auto lane_reduce_max(Array a)
{
    auto result = a[0];
    for(index_int i = 1; i < a.size(); ++i)
        result = result > a[i] ? result : a[i];
    return result;
}

template <class Array>
__device__ auto lane_reduce_min(Array a)
{
    auto result = a[0];
    for(index_int i = 1; i < a.size(); ++i)
        result = result < a[i] ? result : a[i];
    return result;
}

// ============================================================
// DPP (Data Parallel Primitives) wave-level reductions
// ============================================================

// Full wave reduction using butterfly pattern (64 lanes)
template <class T, class Op>
__device__ auto dpp_reduce_impl(T x, Op op)
{
    // Row reduction (within groups of 16)
    x = op(x, __shfl_xor(x, 1));
    x = op(x, __shfl_xor(x, 2));
    x = op(x, __shfl_xor(x, 4));
    x = op(x, __shfl_xor(x, 8));
    // Cross-row reduction
    x = op(x, __shfl_xor(x, 16));
    x = op(x, __shfl_xor(x, 32));
    return x;
}

template <class T>
__device__ auto dpp_reduce_sum(T x)
{
    return dpp_reduce_impl(x, [](auto a, auto b) { return a + b; });
}

template <class T>
__device__ auto dpp_reduce_product(T x)
{
    return dpp_reduce_impl(x, [](auto a, auto b) { return a * b; });
}

template <class T>
__device__ auto dpp_reduce_max(T x)
{
    return dpp_reduce_impl(x, [](auto a, auto b) { return a > b ? a : b; });
}

template <class T>
__device__ auto dpp_reduce_min(T x)
{
    return dpp_reduce_impl(x, [](auto a, auto b) { return a < b ? a : b; });
}

// ============================================================
// Block-level reductions (across wavefronts via LDS)
// ============================================================

template <class T, class Op>
__device__ auto
block_reduce_impl(T x, T* lds, index_int nwaves, index_int wave_id, index_int lane_id, Op op)
{
    // Each wave writes its reduced value to LDS
    if(lane_id == 0)
        lds[wave_id] = x;
    __syncthreads();

    // First wave reduces all partial results
    if(wave_id == 0)
    {
        T val = (lane_id < nwaves) ? lds[lane_id] : x;
        for(index_int stride = 1; stride < nwaves; stride *= 2)
        {
            T other = __shfl_xor(val, stride);
            if(lane_id + stride < nwaves)
                val = op(val, other);
        }
        if(lane_id == 0)
            lds[0] = val;
    }
    __syncthreads();
    return lds[0];
}

template <class T>
__device__ auto
block_reduce_sum(T x, T* lds, index_int nwaves, index_int wave_id, index_int lane_id)
{
    return block_reduce_impl(
        x, lds, nwaves, wave_id, lane_id, [](auto a, auto b) { return a + b; });
}

template <class T>
__device__ auto
block_reduce_max(T x, T* lds, index_int nwaves, index_int wave_id, index_int lane_id)
{
    return block_reduce_impl(
        x, lds, nwaves, wave_id, lane_id, [](auto a, auto b) { return a > b ? a : b; });
}

template <class T>
__device__ auto
block_reduce_min(T x, T* lds, index_int nwaves, index_int wave_id, index_int lane_id)
{
    return block_reduce_impl(
        x, lds, nwaves, wave_id, lane_id, [](auto a, auto b) { return a < b ? a : b; });
}

template <class T>
__device__ auto
block_reduce_product(T x, T* lds, index_int nwaves, index_int wave_id, index_int lane_id)
{
    return block_reduce_impl(
        x, lds, nwaves, wave_id, lane_id, [](auto a, auto b) { return a * b; });
}

// ============================================================
// Index transformation helpers (1D specializations)
// ============================================================

// Pad: transform output index to input index for 1D padding.
// Returns -1 if the output position corresponds to a padded region.
__device__ inline int64_t pad_index_1d(index_int input_len, index_int pad_before, index_int out_idx)
{
    auto in_idx = static_cast<int64_t>(out_idx) - static_cast<int64_t>(pad_before);
    if(in_idx < 0 or in_idx >= static_cast<int64_t>(input_len))
        return -1;
    return in_idx;
}

// Gather: transform output index using an indices tensor for 1D gather on axis 0.
// Returns the data index by looking up indices[out_idx].
template <class Indices>
__device__ auto gather_index_1d(Indices indices, index_int out_idx)
{
    auto idx = static_cast<int64_t>(indices[out_idx]);
    return static_cast<index_int>(idx);
}

// Reverse: transform output index for 1D reversal.
__device__ inline index_int reverse_index_1d(index_int len, index_int out_idx)
{
    return (len - 1) - out_idx;
}

// Conditional load: load from data[idx] if idx is in bounds, else return fill.
template <class T, class Fill>
__device__ auto conditional_load(T data, int64_t idx, Fill fill)
{
    if(idx >= 0 and static_cast<index_int>(idx) < data.get_shape().elements())
        return data[static_cast<index_int>(idx)];
    return static_cast<decltype(data[0])>(fill);
}

} // namespace gen
} // namespace migraphx

#endif // MIGRAPHX_GUARD_KERNELS_GEN_HPP
