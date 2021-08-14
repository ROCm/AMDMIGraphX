
#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_HEAP_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_HEAP_HPP

#include "migraphx/instruction_ref.hpp"
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/device/vector.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <class T, class Index, class Compare>
struct hip_heap
{
    MIGRAPHX_DEVICE_CONSTEXPR hip_heap(T* val, index_int n, Index v_idx, Compare comp)
        : data(val), size(n), data_index(v_idx), compare(comp)
    {
        make_heap(data, size, data_index, compare);
    }

    void update(const T val)
    {
        pop_heap(data, size - 1, data_index, compare);
        if(compare(val, data[data_index(size - 1)]))
        {
            data[data_index(size - 1)] = val;
            push_heap(data, size - 1, data_index, compare);
        }
    }

    void sort() { sort_heap(data, size, data_index, compare); }

    T* get_sorted() { return data; }

    private:
    __device__ inline void swap(T& v1, T& v2)
    {
        T v = v1;
        v1  = v2;
        v2  = v;
    }

    __device__ inline void
    heapify_down(T* ind, index_int n, index_int index, Index ind_idx, Compare comp) // NOLINT
    {
        while(index < n)
        {
            auto pre_index = index;
            index_int l    = 2 * index + 1;
            index_int r    = 2 * index + 2;

            if(l < n && comp(ind[ind_idx(l)], ind[ind_idx(index)]))
            {
                index = l;
            }

            if(r < n && comp(ind[ind_idx(r)], ind[ind_idx(index)]))
            {
                index = r;
                if(comp(ind[ind_idx(l)], ind[ind_idx(r)]))
                {
                    index = l;
                }
            }

            if(index == pre_index)
            {
                break;
            }

            swap(ind[ind_idx(index)], ind[ind_idx(pre_index)]);
        }
    }

    __device__ inline void
    heapify_up(T* ind, index_int index, Index ind_idx, Compare comp) // NOLINT
    {
        while(index > 0)
        {
            auto parent_idx = (index - 1) / 2;

            if(comp(ind[ind_idx(index)], ind[ind_idx(parent_idx)]))
            {
                swap(ind[ind_idx(index)], ind[ind_idx(parent_idx)]);
                index = parent_idx;
            }

            break;
        }
    }

    __device__ inline void make_heap(T* ind, index_int n, Index ind_idx, Compare comp) // NOLINT
    {
        for(index_int j = n / 2 - 1; j >= 0; j--)
        {
            heapify_down(ind, n, j, ind_idx, comp);
        }
    }

    __device__ inline void push_heap(T* ind, index_int loc, Index ind_idx, Compare comp) // NOLINT
    {
        heapify_up(ind, loc, ind_idx, comp);
    }

    __device__ inline void pop_heap(T* ind, index_int loc, Index ind_idx, Compare comp) // NOLINT
    {
        swap(ind[ind_idx(0)], ind[ind_idx(loc)]);
        heapify_down(ind, loc, 0, ind_idx, comp);
    }

    __device__ inline void sort_heap(T* ind, index_int n, Index ind_idx, Compare comp) // NOLINT
    {
        for(int j = n - 1; j > 0; --j)
        {
            swap(ind[ind_idx(0)], ind[ind_idx(j)]);
            heapify_down(ind, j, 0, ind_idx, comp);
        }
    }

    private:
    T* data = nullptr;
    index_int size;
    Index data_index;
    Compare compare;
};

template <class T, class Index, class Compare>
hip_heap<T, Index, Compare> make_heap(T* data, index_int n, Index idx, Compare comp)
{
    return {data, n, idx, comp};
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
