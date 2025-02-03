#ifndef MIGRAPHX_GUARD_KERNELS_TOPK_HPP
#define MIGRAPHX_GUARD_KERNELS_TOPK_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/math.hpp>
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/bit.hpp>
#include <migraphx/kernels/ranges.hpp>
#include <migraphx/kernels/dpp.hpp>
#include <migraphx/kernels/print.hpp>

namespace migraphx {

template<class T>
struct topk_pair
{
    T key;
    uint16_t val;

    template <class Stream>
    friend constexpr const Stream& operator<<(const Stream& ss, const topk_pair& tp)
    {
        ss << "{ " << tp.key << ", " << tp.val << "}";
        return ss;
    }
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

template<class Compare>
struct bitonic_sort
{
    Compare compare_function;

    template<class T, class Reverse>
    constexpr bool compare(const T& x, const T& y, Reverse reverse) const
    {
        if(reverse)
            return compare_function(x, y);
        return compare_function(y, x);
    }

    template<class T>
    constexpr bool compare(const T& x, const T& y) const
    {
        return compare(x, y);
    }

    template<class T, class Reverse>
    constexpr void lane_compare_swap(T& x, T& y, Reverse reverse) const
    {
        if(compare(x, y, reverse))
            swap(x, y);
    }

    template<class GroupSize, class Offset, class Dir, class Array>
    constexpr void lane_shuffle(GroupSize group_size, Offset offset, Dir dir, Array& x) const
    {
        MIGRAPHX_ASSERT(is_power_of_2(x.size()));
        MIGRAPHX_ASSERT(is_power_of_2(group_size));
        MIGRAPHX_ASSERT(is_power_of_2(offset));
        if constexpr(group_size >= 2 and offset >= 1)
        {
            constexpr auto step = _c<2> * offset;
            repeat(x.size() / step, [&](auto q) {
                auto base = q * step;

                // The local direction must change every group_size items
                // and is flipped if dir is true
                const auto local_dir = ((base & group_size) > _c<0>) != dir;

                for(index_int i = 0; i < offset; i++)
                    lane_compare_swap(x[base + i], x[base + i + offset], local_dir);
            });
            if constexpr(offset > 1)
                lane_shuffle(group_size, offset / _c<2>, dir, x);
        }
    }

    template<class Dir, class Array>
    constexpr void lane_merge(Dir dir, Array& x) const
    {
        if constexpr(decltype(x.size()){} < 2)
            return;
        lane_shuffle(x.size(), x.size() / _c<2>, dir, x);
    }

    template<class K, class Dir, class Array>
    constexpr void lane_sort(K k, Dir dir, Array& x) const
    {
        lane_shuffle(k, k / _c<2>, dir, x);
        if constexpr(k < decltype(x.size()){})
            lane_sort(k * _c<2>, dir, x);
    }

    template<class Dir, class Array>
    constexpr void lane_sort(Dir dir, Array& x) const
    {
        lane_sort(_c<2>, dir, x);
    }

    template<class Mask, class Dir, class Array>
    __device__ void dpp_swap(Mask mask, Dir dir, Array& x) const
    {
        // println_once("swap: ", mask);
        MIGRAPHX_ASSERT(mask > 0);
        MIGRAPHX_ASSERT(mask < MIGRAPHX_WAVEFRONTSIZE);
        repeat(x.size(), [&](auto item) {
            auto& src = x[item];
            auto partner = readlane_xor<mask>(src);
            if(compare(src, partner, dir))
                src = partner;
        });
    }

    template<class Array>
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
                auto j = w - i - _c<1>;
                auto mask = _c<1u> << j; // pow(2, j)
                dpp_swap(mask, get_bit(id, w) != get_bit(id, j), x);
            });
            if constexpr(w == 0)
                lane_sort(get_bit(id, w) != 0, x);
            else
                lane_merge(get_bit(id, w) != 0, x);
        });
    }
};

template <class Compare>
bitonic_sort(Compare) -> bitonic_sort<Compare>;

// template<index_int N, class T, class Compare>
// __device__ void bitonic_sort(index idx, T* buf, Compare compare)
// {
//     powers_of_2(N)([&](auto k) {
//         auto dir = k*2;
//         reverse_powers_of_2(k)([&](auto j) {
//             idx.local_stride(N, [&](auto i) {
//                 auto ij = i ^ j;
//                 if(ij <= i)
//                     return;
//                 MIGRAPHX_ASSERT(ij < N);
//                 bool reverse = (i & dir) == 0;
//                 if(reverse ^ compare(buf[i], buf[ij]))
//                     swap(buf[ij], buf[i]);
//             });
//             __syncthreads();
//         });
//     });
// }

template<index_int N, index_int K, class Compare>
struct bitonic_topk
{
    Compare compare;

    static_assert(K < N, "K must be less than N");

    // Constructor used to enable deduction guidelines
    constexpr bitonic_topk(index_constant<N>,
                           index_constant<K>,
                           Compare cmp)
      : compare(cmp)
    { }

    template<class T>
    __device__ void sort_step(index idx, T* buf, index_int len) const
    {
        auto dir = len * 2;
        reverse_powers_of_2(len)([&](auto inc) {
            idx.local_stride(N, [&](auto tid) {
                auto low = tid & (inc - 1);
                auto i = (tid * 2) - low;
                auto j = i + inc;
                if(j >= N)
                    return;
                MIGRAPHX_ASSERT(i < N);
                MIGRAPHX_ASSERT(j < N);
                bool reverse = (dir & i) == 0;
                if(reverse ^ compare(buf[i], buf[j]))
                    swap(buf[i], buf[j]);
            });
            __syncthreads();
        });
    }

    template<class T>
    __device__ void merge_step(index idx, T* buf, index_int len) const
    {
        auto dir = len * 2;
        idx.local_stride(N, [&](auto tid) {
            auto low = tid & (len - 1);
            auto i = (tid * 2) - low;
            auto j = i + len;
            if(j >= N)
                return;
            if (i % dir >= K)
                return;
            MIGRAPHX_ASSERT(i < N);
            buf[i] = min(buf[i], buf[j], compare);
        });
        __syncthreads();
    }

    template<class T>
    __device__ void rebuild_step(index idx, T* buf, index_int len) const
    {
        auto dir = len * 2;
        reverse_powers_of_2(K / 2)([&](auto inc) {
            idx.local_stride(N, [&](auto tid) {
                auto low = tid & (inc - 1);
                auto i = (tid * 2) - low;
                auto j = i + inc;
                if(j >= N)
                    return;
                MIGRAPHX_ASSERT(i < N);
                if (i % dir >= K)
                    return;
                bool reverse = (dir & i) == 0;
                if(reverse ^ compare(buf[i], buf[j]))
                    swap(buf[i], buf[j]);
            });
            __syncthreads();
        });
    }

    template<class T>
    __device__ void block_topk(index idx, T* buf) const
    {
        powers_of_2(K)([&](auto len) {
            sort_step(idx, buf, len);
        });
        powers_of_2(K, N)([&](auto len) {
            merge_step(idx, buf, len);
            rebuild_step(idx, buf, len);
        });
    }

};

template <class N, class K, class Compare>
bitonic_topk(N, K, Compare) -> bitonic_topk<N{}, K{}, Compare>;

template<index_int Axis, class Output, class Indices, class Input, class Compare, class T>
__device__ void topk(Output output, Indices indices, Input input, Compare compare, T init)
{
    using type = typename Input::type;
    auto idx = make_index();
    constexpr auto n = return_c([] { return get_shape_c<Input>{}.get_shape().lens[Axis]; });
    constexpr auto k = return_c([] { return get_shape_c<Output>{}.get_shape().lens[Axis]; });
    idx.group_stride(input.get_shape().elements() / n, [&](auto group) {
        auto x = tensor_slice(input, group, slice_axes<Axis>());
        auto y = tensor_slice(output, group, slice_axes<Axis>());
        auto y_idx = tensor_slice(indices, group, slice_axes<Axis>());
#if 1
        constexpr auto nlocal_wave = idx.nlocal_wave();
        constexpr auto per_lane = return_c([=] { return bit_ceil(n / nlocal_wave); });
        array<topk_pair<type>, per_lane> local_buf;
        const auto base = idx.local_wave() * per_lane;
        // copy to registers
        for(index_int i:range(per_lane))
        {
            auto j = i + base;
            local_buf[i].key = j < n ? x[j] : init;
            local_buf[i].val = j;
        }
        // println(base, ": ", local_buf);

        // Deduction guide is broken for some reason
        // bitonic_sort{by(select_key(), compare)}.wave_sort(idx, local_buf);
        auto c = by(select_key(), compare);
        bitonic_sort<decltype(c)>{c}.wave_sort(idx, local_buf);

        // Copy to output
        for(index_int i:range(per_lane))
        {
            auto j = i + base;
            if(j >= k)
                continue;
            y[j] = local_buf[i].key;
            y_idx[j] = local_buf[i].val;
        }

#else
        constexpr auto aligned_k = return_c([=] { return bit_ceil(k); });
        constexpr auto aligned_n = return_c([=] { return bit_ceil(n); });
        __shared__ topk_pair<type> buf[aligned_n];
        // Copy to LDS
        idx.local_stride(aligned_n, [&](auto i) {
            auto key = i < x.get_shape().elements() ? x[i] : init;
            buf[i].key = key;
            buf[i].val = i;
        });
        __syncthreads();
#if 1
        bitonic_topk{aligned_n, aligned_k, by(select_key(), compare)}.block_topk(idx, buf);

        // powers_of_2(aligned_k)([&](auto len) {
        //     bitonic_topk_sort_step(idx, buf, aligned_n, len, by(select_key(), compare));
        // });
        // powers_of_2(aligned_k, aligned_n)([&](auto len) {
        //     bitonic_topk_merge_step<aligned_k>(idx, buf, aligned_n, len, by(select_key(), compare));
        //     bitonic_topk_rebuild_step<aligned_k>(idx, buf, aligned_n, len, by(select_key(), compare));
        // });
#elif 1
        // bitonic_sort<aligned_n>(idx, buf, by(select_key(), compare));
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
#endif
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_TOPK_HPP

