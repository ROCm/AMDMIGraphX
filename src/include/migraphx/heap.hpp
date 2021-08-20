#ifndef MIGRAPHX_GUARD_RTGLIB_HEAP_HPP
#define MIGRAPHX_GUARD_RTGLIB_HEAP_HPP

#include <thread>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cassert>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class T, class Compare>
struct heap
{
    std::vector<T> data;
    Compare compare;

    heap(const std::vector<T>& val, Compare comp) : data(val), compare(std::move(comp))
    {
        for(int i = 2; i <= data.size(); ++i)
        {
            std::push_heap(data.begin(), data.begin() + i, compare);
        }
    }

    void update(const T val)
    {
        std::pop_heap(data.begin(), data.end(), compare);
        if(compare(val, data.back()))
            data.back() = val;
        std::push_heap(data.begin(), data.end(), compare);
    }

    void sort() { std::sort_heap(data.begin(), data.end(), compare); }

    std::vector<T>& get() { return data; }
};

template <class T, class Compare>
heap<T, Compare> make_heap(std::vector<T>& val, Compare compare)
{
    return {std::move(val), std::move(compare)};
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
