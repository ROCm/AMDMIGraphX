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
#ifndef MIGRAPHX_GUARD_KERNELS_SORT_HPP
#define MIGRAPHX_GUARD_KERNELS_SORT_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/math.hpp>
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/bit.hpp>
#include <migraphx/kernels/ranges.hpp>
#include <migraphx/kernels/dpp.hpp>
#include <migraphx/kernels/type_traits.hpp>

namespace migraphx {

template <class Compare>
struct bitonic_sort
{
    Compare compare_function;

    template <class T, class Reverse>
    constexpr bool compare(const T& x, const T& y, Reverse reverse) const
    {
        return reverse ^ compare_function(y, x);
    }

    template <class T>
    constexpr bool compare(const T& x, const T& y) const
    {
        return compare(x, y);
    }

    template <class T, class Reverse>
    constexpr void lane_compare_swap(T& x, T& y, Reverse reverse) const
    {
        if(compare(x, y, reverse))
            swap(x, y);
    }

    template <class Reverse>
    constexpr auto compare(Reverse reverse) const
    {
        return [=](const auto& x, const auto& y) { return compare(x, y, reverse); };
    }

    template <class GroupSize, class Dir, class Array>
    constexpr void lane_shuffle(GroupSize group_size, Dir dir, Array& x) const
    {
        MIGRAPHX_ASSERT(is_power_of_2(x.size()));
        MIGRAPHX_ASSERT(is_power_of_2(group_size));
        if constexpr(group_size >= 2)
        {
            repeat_down_by_2_c<group_size / 2>([&](auto offset) {
                constexpr auto step = _c<2 * offset>;
                repeat(x.size() / step, [&](auto q) {
                    auto base = q * step;

                    // The local direction must change every group_size items
                    // and is flipped if dir is true
                    const auto local_dir = ((base & group_size) > _c<0>) != dir;

                    repeat(offset, [&](auto i) {
                        lane_compare_swap(x[base + i], x[base + i + offset], local_dir);
                    });
                });
            });
        }
    }

    template <class Dir, class Array>
    constexpr void lane_merge(Dir dir, Array& x) const
    {
        if constexpr(decltype(x.size()){} < 2)
            return;
        lane_shuffle(x.size(), dir, x);
    }

    template <class Dir, class Array>
    constexpr void lane_sort(Dir dir, Array& x) const
    {
        repeat_up_by_2_c<2, decltype(x.size()){} * 2>([&](auto k) { lane_shuffle(k, dir, x); });
    }

    template <class Mask, class Dir, class Array>
    __device__ void dpp_swap(Mask mask, Dir dir, Array& x) const
    {
        MIGRAPHX_ASSERT(mask > 0);
        MIGRAPHX_ASSERT(mask < MIGRAPHX_WAVEFRONTSIZE);
        repeat(x.size(), [&](auto item) {
            auto& src    = x[item];
            auto partner = readlane_xor<mask>(src);
            if(compare(src, partner, dir))
                src = partner;
        });
    }

    template <class Array>
    __device__ void wave_sort(index idx, Array& x) const
    {
        static_assert(is_power_of_2(decltype(x.size()){}), "Array size must be power of 2");
#if MIGRAPHX_WAVEFRONTSIZE == 64
        constexpr auto max_width = _c<6>;
#else
        constexpr auto max_width = _c<5>;
#endif

        const auto id = idx.local_wave();
        repeat(max_width + _c<1>, [&](auto w) {
            repeat(w, [&](auto i) {
                auto j    = w - i - _c<1>;
                auto mask = _c<1u> << j; // pow(2, j)
                dpp_swap(mask, get_bit(id, w) != get_bit(id, j), x);
            });
            if constexpr(w == 0)
                lane_sort(get_bit(id, w), x);
            else
                lane_merge(get_bit(id, w), x);
        });
    }

    // Block-wide bitonic sort of N-element buffer (N must be a power of 2)
    // with caller-supplied swap function.
    // Used when multiple parallel arrays must be swapped in lockstep.
    // compare_function(i, j) is true if position j should sort before
    // position i.
    template <index_int N, class SwapAt>
    __device__ void block_sort_by_index(index idx, SwapAt swap_at) const
    {
        static_assert(is_power_of_2(N), "N must be a power of 2");
        repeat_up_by_2_c<2, (N * 2)>([&](auto k) {
            repeat_down_by_2_c<(k >> 1)>([&](auto j) {
                idx.local_stride(N, [&](auto tid) {
                    index_int partner = tid ^ j;
                    if(partner > tid)
                    {
                        const bool reverse = (tid & k) != 0;
                        if(reverse ^ compare_function(tid, partner))
                            swap_at(tid, partner);
                    }
                });
                __syncthreads();
            });
        });
    }
};

MIGRAPHX_AUTO_DEDUCE(bitonic_sort);

template <index_int N, index_int K, class Compare>
struct bitonic_topk
{
    Compare compare_function;

    static_assert(K <= N, "K must be less than N");

    // Constructor used to enable deduction guidelines
    constexpr bitonic_topk(index_constant<N>, index_constant<K>, Compare cmp)
        : compare_function(cmp)
    {
    }

    template <class T, class Reverse>
    constexpr bool compare(const T& x, const T& y, Reverse reverse) const
    {
        return reverse ^ compare_function(x, y);
    }

    template <class T, class Len>
    __device__ void sort_step(index idx, T* buf, Len len) const
    {
        auto dir = len * 2;
        repeat_down_by_2_c<len>([&](auto inc) {
            idx.local_stride(N, [&](auto tid) {
                auto low = tid & (inc - 1);
                auto i   = (tid * 2) - low;
                auto j   = i + inc;
                if(j >= N)
                    return;
                MIGRAPHX_ASSERT(i < N);
                MIGRAPHX_ASSERT(j < N);
                bool reverse = (dir & i) == 0;
                if(compare(buf[i], buf[j], reverse))
                    swap(buf[i], buf[j]);
            });
            __syncthreads();
        });
    }

    template <class T, class Len>
    __device__ void merge_step(index idx, T* buf, Len len) const
    {
        auto dir = len * 2;
        idx.local_stride(N, [&](auto tid) {
            auto low = tid & (len - 1);
            auto i   = (tid * 2) - low;
            auto j   = i + len;
            if(j >= N)
                return;
            if(i % dir >= K)
                return;
            MIGRAPHX_ASSERT(i < N);
            buf[i] = min(buf[i], buf[j], compare_function);
        });
        __syncthreads();
    }

    template <class T, class Len>
    __device__ void rebuild_step(index idx, T* buf, Len len) const
    {
        auto dir = len * 2;
        repeat_down_by_2_c<K / 2>([&](auto inc) {
            idx.local_stride(N, [&](auto tid) {
                auto low = tid & (inc - 1);
                auto i   = (tid * 2) - low;
                auto j   = i + inc;
                if(j >= N)
                    return;
                MIGRAPHX_ASSERT(i < N);
                if(i % dir >= K)
                    return;
                bool reverse = (dir & i) == 0;
                if(compare(buf[i], buf[j], reverse))
                    swap(buf[i], buf[j]);
            });
            __syncthreads();
        });
    }

    template <class T>
    __device__ void block_topk(index idx, T* buf) const
    {
        repeat_up_by_2_c<K>([&](auto len) { sort_step(idx, buf, len); });
        repeat_up_by_2_c<K, N>([&](auto len) {
            merge_step(idx, buf, len);
            rebuild_step(idx, buf, len);
        });
    }
};

template <class N, class K, class Compare>
bitonic_topk(N, K, Compare) -> bitonic_topk<N{}, K{}, Compare>;

template <class T, class U>
struct topk_pair_t_u
{
    T key;
    U val;
};

template <class T, class U>
struct topk_pair_u_t
{
    U val;
    T key;
};

/// Key and index of a topk candidate, with the larger member first to avoid padding
template <class T, class U>
struct topk_pair : conditional_t<(sizeof(T) >= sizeof(U)), topk_pair_t_u<T, U>, topk_pair_u_t<T, U>>
{
    template <class Stream>
    friend constexpr const Stream& operator<<(const Stream& ss, const topk_pair& tp)
    {
        ss << "{ " << tp.key << ", " << tp.val << "}";
        return ss;
    }
};

template <class Pair, class T, class U>
constexpr Pair make_topk_pair(T key, U val)
{
    Pair p;
    p.key = key;
    p.val = val;
    return p;
}

/// Orders the pairs by key, with the lower index first among equal keys
template <class Compare>
constexpr auto compare_topk_pair(Compare compare)
{
    return [=](const auto& x, const auto& y) {
        if(compare(x.key, y.key))
            return true;
        if(compare(y.key, x.key))
            return false;
        return x.val < y.val;
    };
}

/// Selects the top K of the n elements in a workgroup. read(j, d) returns the
/// topk_pair of element j in local_stride order, and the sorted top K are
/// passed to write(i, d, p) in local_stride order. Each wave sorts its
/// elements in registers and the per-wave candidates are merged through lds,
/// unless there would be as many candidates as elements, in which case all
/// the elements are merged through lds directly.
template <index_int K, class Compare, class T, class N, class Read, class Write>
__device__ void select_topk(index idx, Compare compare, T init, N, Read read, Write write)
{
    using pair                   = decltype(read(index_int{0}, _c<0>));
    constexpr auto n             = N{};
    constexpr auto k             = _c<K>;
    constexpr auto aligned_n     = _c<bit_ceil(n)>;
    constexpr auto aligned_k     = _c<bit_ceil(K)>;
    constexpr auto nwave         = idx.nwave();
    constexpr auto m             = k * nwave;
    constexpr auto aligned_m     = _c<bit_ceil(m)>;
    constexpr bool wave_select   = aligned_m < aligned_n or nwave == 1;
    constexpr index_int buf_size = wave_select ? aligned_m : aligned_n;
    static_assert(K <= n, "K must not be larger than n");
    const auto sentinel = make_topk_pair<pair>(init, -1);
    __shared__ pair buf[buf_size];
    // Wait for any previous selection to finish reading the buffer
    __syncthreads();
    if constexpr(wave_select)
    {
        constexpr auto nper_lane = _c<bit_ceil(decltype(idx.max_local_stride_iterations(n)){})>;
        array<pair, nper_lane> local_buf;
        for(index_int i : range(nper_lane))
            local_buf[i] = sentinel;
        idx.local_stride(n, [&](auto j, auto d) { local_buf[d] = read(j, d); });
        bitonic_sort{compare_topk_pair(compare)}.wave_sort(idx, local_buf);
        // Each wave keeps its top K candidates
        const auto base = idx.local_wave() * nper_lane;
        for(index_int i : range(nper_lane))
        {
            auto ibase = i + base;
            if(ibase >= k)
                continue;
            buf[idx.wave() * k + ibase] = local_buf[i];
        }
        if constexpr(nwave > 1)
        {
            idx.local_stride(aligned_m - m, [&](auto i) { buf[m + i] = sentinel; });
            __syncthreads();
            bitonic_topk{aligned_m, aligned_k, compare_topk_pair(compare)}.block_topk(idx, buf);
        }
        else
        {
            __syncthreads();
        }
    }
    else
    {
        idx.local_stride(n, [&](auto j, auto d) { buf[j] = read(j, d); });
        idx.local_stride(aligned_n - n, [&](auto i) { buf[n + i] = sentinel; });
        __syncthreads();
        bitonic_topk{aligned_n, aligned_k, compare_topk_pair(compare)}.block_topk(idx, buf);
    }
    idx.local_stride(k, [&](auto i, auto d) { write(i, d, buf[i]); });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_SORT_HPP
