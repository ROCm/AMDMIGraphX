#ifndef MIGRAPHX_GUARD_KERNELS_TOPK_HPP
#define MIGRAPHX_GUARD_KERNELS_TOPK_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/math.hpp>
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/bit.hpp>
#include <migraphx/kernels/print.hpp>

namespace migraphx {

template<class T>
struct topk_pair
{
    T key;
    uint16_t val;
};

constexpr auto select_key()
{
    return [](const auto& p) { return p.key; };
}

constexpr auto powers_of_2(index_int start, index_int last)
{
    return [=](auto f) {
        for(index_int i=start;i<last;i*=2)
            f(i);
    };
}
constexpr auto powers_of_2(index_int last)
{
    return powers_of_2(1, last);
}

constexpr auto reverse_powers_of_2(index_int start, index_int last = 1)
{
    return [=](auto f) {
        for(index_int i=start;i>=last;i/=2)
            f(i);
    };
}

template<index_int N, class T, class Compare>
__device__ void bitonic_sort(index idx, T* buf, Compare compare)
{
    powers_of_2(N)([&](auto k) {
        auto dir = k*2;
        reverse_powers_of_2(k)([&](auto j) {
            idx.local_stride(N, [&](auto i) {
                auto ij = i ^ j;
                if(ij <= i)
                    return;
                MIGRAPHX_ASSERT(ij < N);
                bool reverse = (i & dir) == 0;
                if((reverse and compare(buf[ij], buf[i])) or (not reverse and compare(buf[i], buf[ij])))
                    swap(buf[ij], buf[i]);
            });
            __syncthreads();
        });
    });
}

template<class N, class T, class Compare>
__device__ void bitonic_topk_sort_step(index idx, T* buf, N n, index_int len, Compare compare)
{
    auto dir = len * 2;
    reverse_powers_of_2(len)([&](auto inc) {
        idx.local_stride(n, [&](auto tid) {
            auto low = tid & (inc - 1);
            auto i = (tid * 2) - low;
            auto j = i + inc;
            if(j >= n)
                return;
            MIGRAPHX_ASSERT(i < n);
            MIGRAPHX_ASSERT(j < n);
            bool reverse = (dir & i) == 0;
            if(reverse ^ compare(buf[i], buf[j]))
                swap(buf[i], buf[j]);
        });
        __syncthreads();
    });
}

template<index_int K, class N, class T, class Compare>
__device__ void bitonic_topk_merge_step(index idx, T* buf, N n, index_int len, Compare compare)
{
    auto dir = len * 2;
    idx.local_stride(n, [&](auto tid) {
        auto low = tid & (len - 1);
        auto i = (tid * 2) - low;
        auto j = i + len;
        if(j >= n)
            return;
        if (i % dir >= K)
            return;
        MIGRAPHX_ASSERT(i < n);
        buf[i] = min(buf[i], buf[j], compare);
    });
    __syncthreads();
}

template<index_int K, class N, class T, class Compare>
__device__ void bitonic_topk_rebuild_step(index idx, T* buf, N n, index_int len, Compare compare)
{
    auto dir = len * 2;
    reverse_powers_of_2(K / 2)([&](auto inc) {
        idx.local_stride(n, [&](auto tid) {
            auto low = tid & (inc - 1);
            auto i = (tid * 2) - low;
            auto j = i + inc;
            if(j >= n)
                return;
            MIGRAPHX_ASSERT(i < n);
            if (i % dir >= K)
                return;
            bool reverse = (dir & i) == 0;
            if(reverse ^ compare(buf[i], buf[j]))
                swap(buf[i], buf[j]);
        });
        __syncthreads();
    });
}

template<index_int Axis, class Output, class Indices, class Input, class Compare, class T>
__device__ void topk(Output output, Indices indices, Input input, Compare compare, T init)
{
    using type = typename Input::type;
    auto idx = make_index();
    constexpr auto n = return_c([] { return get_shape_c<Input>{}.get_shape().lens[Axis]; });
    constexpr auto k = return_c([] { return get_shape_c<Output>{}.get_shape().lens[Axis]; });
    constexpr auto aligned_k = return_c([=] { return bit_ceil(k); });
    constexpr auto aligned_n = return_c([=] { return bit_ceil(n); });
    __shared__ topk_pair<type> buf[aligned_n];
    idx.group_stride(input.get_shape().elements() / n, [&](auto group) {
        auto x = tensor_slice(input, group, slice_axes<Axis>());
        auto y = tensor_slice(output, group, slice_axes<Axis>());
        auto y_idx = tensor_slice(indices, group, slice_axes<Axis>());
        // Copy to LDS
        idx.local_stride(aligned_n, [&](auto i) {
            auto key = i < x.get_shape().elements() ? x[i] : init;
            buf[i].key = key;
            buf[i].val = i;
        });
        __syncthreads();
#if 1
        powers_of_2(aligned_k)([&](auto len) {
            bitonic_topk_sort_step(idx, buf, aligned_n, len, by(select_key(), compare));
        });
        powers_of_2(aligned_k, aligned_n)([&](auto len) {
            bitonic_topk_merge_step<aligned_k>(idx, buf, aligned_n, len, by(select_key(), compare));
            bitonic_topk_rebuild_step<aligned_k>(idx, buf, aligned_n, len, by(select_key(), compare));
        });
#elif 0
        bitonic_sort<aligned_n>(idx, buf, by(select_key(), compare));
#else
        auto c = by(select_key(), compare);
        auto bswap = [&](auto dir, auto i, auto j) {
            bool reverse = (dir & i) == 0;
            if(reverse ^ c(buf[i], buf[j]))
                swap(buf[i], buf[j]);
        };
        // sort each K
        powers_of_2(aligned_k)([&](auto len) {
            auto dir = len * 2;
            reverse_powers_of_2(len)([&](auto inc) {
                idx.local_stride(aligned_n, [&](auto tid) {
                    auto low = tid & (inc - 1);
                    auto i = (tid * 2) - low;
                    auto j = i + inc;
                    if(j >= aligned_n)
                        return;
                    bswap(dir, i, j);
                });
                __syncthreads();
            });
        });
        // merge and rebuild K
        powers_of_2(aligned_k, aligned_n)([&](auto len) {
            auto dir = len * 2;
            idx.local_stride(aligned_n, [&](auto tid) {
                auto i = (tid * 2) - (tid & (len - 1));
                auto j = i + len;
                if (i % dir < aligned_k and j < aligned_n)
                {
                    buf[i] = min(buf[i], buf[j], c);
                }
            });
            __syncthreads();
            reverse_powers_of_2(aligned_k / 2)([&](auto inc) {
                idx.local_stride(aligned_n, [&](auto tid) {
                    auto low = tid & (inc - 1);
                    auto ii = (tid * 2) - low;
                    auto jj = ii + inc;
                    if (ii % dir < aligned_k and jj < aligned_n)
                        bswap(dir, ii, jj);
                });
                __syncthreads();
            });
        });
#endif
        // save top K
        idx.local_stride(aligned_k, [&](auto i) {
            y[i] = buf[i].key;
            y_idx[i] = buf[i].val;
        });
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_TOPK_HPP

