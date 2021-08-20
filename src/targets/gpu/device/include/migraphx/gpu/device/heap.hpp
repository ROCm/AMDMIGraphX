
#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_HEAP_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_HEAP_HPP

#include <migraphx/gpu/device/types.hpp>

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
        make_heap(size);
    }

    MIGRAPHX_DEVICE_CONSTEXPR void update(const T val)
    {
        pop_heap(size - 1);
        if(compare(val, data[data_index(size - 1)]))
        {
            return;
        }
        data[data_index(size - 1)] = val;
        push_heap(size - 1);
    }

    MIGRAPHX_DEVICE_CONSTEXPR void sort() { sort_heap(size); }

    MIGRAPHX_DEVICE_CONSTEXPR T* get() { return data; }

    private:
    MIGRAPHX_DEVICE_CONSTEXPR inline static void swap(T& v1, T& v2)
    {
        T v = v1;
        v1  = v2;
        v2  = v;
    }

    MIGRAPHX_DEVICE_CONSTEXPR inline void heapify_down(index_int n, index_int index)
    {
        while(index < n)
        {
            auto pre_index = index;
            index_int l    = 2 * index + 1;
            index_int r    = 2 * index + 2;

            if(l < n && compare(data[data_index(l)], data[data_index(index)]))
            {
                index = l;
            }

            if(r < n && compare(data[data_index(r)], data[data_index(index)]))
            {
                index = r;
                if(compare(data[data_index(l)], data[data_index(r)]))
                {
                    index = l;
                }
            }

            if(index == pre_index)
            {
                break;
            }

            swap(data[data_index(index)], data[data_index(pre_index)]);
        }
    }

    MIGRAPHX_DEVICE_CONSTEXPR inline void heapify_up(index_int index)
    {
        while(index > 0)
        {
            auto parent_idx = (index - 1) / 2;

            if(not compare(data[data_index(index)], data[data_index(parent_idx)]))
            {
                break;
            }

            swap(data[data_index(index)], data[data_index(parent_idx)]);
            index = parent_idx;
        }
    }

    MIGRAPHX_DEVICE_CONSTEXPR inline void make_heap(index_int n)
    {
        for(int j = 1; j < n; ++j)
        {
            heapify_up(j);
        }
    }

    MIGRAPHX_DEVICE_CONSTEXPR inline void push_heap(index_int loc) { heapify_up(loc); }

    MIGRAPHX_DEVICE_CONSTEXPR inline void pop_heap(index_int loc)
    {
        swap(data[data_index(0)], data[data_index(loc)]);
        heapify_down(loc, 0);
    }

    MIGRAPHX_DEVICE_CONSTEXPR inline void sort_heap(index_int n)
    {
        for(int j = n - 1; j > 0; --j)
        {
            swap(data[data_index(0)], data[data_index(j)]);
            heapify_down(j, 0);
        }
    }

    T* data = nullptr;
    index_int size;
    Index data_index;
    Compare compare;
};

template <class T, class Index, class Compare>
__device__ hip_heap<T, Index, Compare> make_heap(T* data, index_int n, Index idx, Compare compare)
{
    return {data, n, idx, compare};
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
