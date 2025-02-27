#ifndef MIGRAPHX_GUARD_KERNELS_SORT_HPP
#define MIGRAPHX_GUARD_KERNELS_SORT_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/math.hpp>
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/bit.hpp>
#include <migraphx/kernels/ranges.hpp>
#include <migraphx/kernels/dpp.hpp>

namespace migraphx {

constexpr auto powers_of_2(index_int start, index_int last)
{
    return [=](auto f) {
        for(index_int i = start; i < last; i *= 2)
            f(i);
    };
}
constexpr auto powers_of_2(index_int last) { return powers_of_2(1, last); }

constexpr auto reverse_powers_of_2(index_int start, index_int last = 1)
{
    return [=](auto f) {
        for(index_int i = start; i >= last; i /= 2)
            f(i);
    };
}

template<index_int Start, index_int Last, class F>
constexpr void repeat_up_by_2_c(F&& f)
{
    if constexpr(Start < Last)
    {
        f(_c<Start>);
        repeat_up_by_2_c<Start * 2, Last>(static_cast<F&&>(f));
    }
}

template<index_int Last, class F>
constexpr void repeat_up_by_2_c(F&& f)
{
    repeat_up_by_2_c<1, Last>(static_cast<F&&>(f));
}

template<index_int Start, index_int Last, class F>
constexpr void repeat_down_by_2_c(F&& f)
{
    if constexpr(Start >= Last)
    {
        f(_c<Start>);
        repeat_down_by_2_c<Start / 2, Last>(static_cast<F&&>(f));
    }
}
template<index_int Start, class F>
constexpr void repeat_down_by_2_c(F&& f)
{
    repeat_down_by_2_c<Start, 1>(static_cast<F&&>(f));
}

template <class Compare>
struct bitonic_sort
{
    Compare compare_function;

    template <class T, class Reverse>
    constexpr bool compare(const T& x, const T& y, Reverse reverse) const
    {
        if(reverse)
            return compare_function(x, y);
        return compare_function(y, x);
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
            repeat_down_by_2_c<group_size/2>([&](auto offset) {
                constexpr auto step = _c<2> * offset;
                repeat(x.size() / step, [&](auto q) {
                    auto base = q * step;

                    // The local direction must change every group_size items
                    // and is flipped if dir is true
                    const auto local_dir = ((base & group_size) > _c<0>) != dir;

                    for(index_int i = 0; i < offset; i++)
                        lane_compare_swap(x[base + i], x[base + i + offset], local_dir);
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
        repeat_up_by_2_c<2, decltype(x.size()){} * 2>([&](auto k) {
            lane_shuffle(k, dir, x);
        });
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
};

MIGRAPHX_AUTO_DEDUCE(bitonic_sort);

template <index_int N, index_int K, class Compare>
struct bitonic_topk
{
    Compare compare;

    static_assert(K <= N, "K must be less than N");

    // Constructor used to enable deduction guidelines
    constexpr bitonic_topk(index_constant<N>, index_constant<K>, Compare cmp) : compare(cmp) {}

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
                if(reverse ^ compare(buf[i], buf[j]))
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
            buf[i] = min(buf[i], buf[j], compare);
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
                if(reverse ^ compare(buf[i], buf[j]))
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

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_SORT_HPP

